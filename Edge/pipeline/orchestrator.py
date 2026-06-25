import time
import cv2
import numpy as np
import os
import socket
import queue
import threading
import requests
from datetime import datetime
from pathlib import Path

from Alertas.database_handler import DatabaseHandler
from bytetracker import BYTETracker
from Edge.Detecao.detector_factory import create_detector
from Edge.Detecao.skeleton import LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ELBOW, RIGHT_ELBOW
from .plugins import load_activities, load_alert_handlers
from .metrics import PipelineMetrics
from .renderer import PoseRenderer
from .video_source import create_video_source
from .spatial_normalizer import SpatialNormalizer
from .skeleton_visualizer import SkeletonVisualizer
from .temporal_pose_filter import TemporalPoseFilter


class TorchTensorMock(np.ndarray):
    """
    Mock class for PyTorch Tensors when interfacing with BYTETracker.
    Avoids importing torch on ONNX-only runs by subclassing np.ndarray
    and mimicking expected PyTorch Tensor methods (.numpy(), .clone()).
    """
    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=np.float32).view(cls)

    def numpy(self):
        return np.asarray(self)

    def clone(self):
        return self.copy()


class AntiTheftOrchestrator:
    """Coordena detector, atividades, alertas, render e loop de video."""

    def __init__(self, config):
        self.config = config
        self.runtime_config = config.runtime()

        # Inicializa o detector de pose
        self.detector = create_detector(config.detector_config())
        self.detector_info = self.detector.get_info() or {}

        self.activities = load_activities(config.activity_specs())
        self.alert_handlers = load_alert_handlers(config.alert_specs())

        self.renderer = PoseRenderer(config.visualization())
        self.video_source = create_video_source(config.camera())
        self.metrics = PipelineMetrics(frame_skip=config.frame_skip())

        self.cache_result = bool(self.runtime_config.get("cache_result", True))
        self.debug = bool(self.runtime_config.get("debug", False))

        # Filtro temporal: o objeto raiz serve apenas como config/toggle.
        # Filtros por track_id estão em temporal_filters_por_id.
        self.temporal_filter = TemporalPoseFilter(config)
        self.temporal_filters_por_id = {}

        # Normalizador espacial (sempre ativo — necessário para os detectores de atividade)
        self.normalizer = SpatialNormalizer(config)

        # show_normalized_window controla apenas a janela extra de debug normalizado
        spatial_cfg = config.data.get("spatial_normalization", {})
        self.show_normalized_window = spatial_cfg.get("enabled", False)
        self.skeleton_viz = SkeletonVisualizer() if self.show_normalized_window else None

        self.last_detection = ([], [])
        self.last_inference_ms = 0.0

        tracker_cfg = config.tracker()
        self.tracker = BYTETracker(
            track_thresh=float(tracker_cfg.get("track_thresh", 0.35)),
            track_buffer=int(tracker_cfg.get("track_buffer", 90)),
            match_thresh=float(tracker_cfg.get("match_thresh", 0.7)),
            frame_rate=int(config.camera().get("fps", 30)),
        )
        self.last_tracked_objects = []

        self.db = DatabaseHandler()
        self._debug_bbox_printed = False

        # Alert persistence fields
        self.current_alert_text = None
        self.current_alert_countdown = 0
        self.alert_persist_frames = 45  # ~1.5s at 30fps

        # Configuração de guarda de métricas
        self.metricas_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Metricas')
        os.makedirs(self.metricas_dir, exist_ok=True)
        self.node_id = self._resolve_node_id()
        self.metricas_intervalo = 300  # a cada 300 frames (~10s a 30fps)
        self.ultimo_frame_metricas = 0

        # Frame web para stream
        # TODO: LIMPAR ESTE CÓDIGO E REMOVER ESTAS VARIAVEIS
        self.frame_web_intervalo = 10
        self.ultimo_frame_web = 0
        self.frame_web_path = os.path.join(self.metricas_dir, 'last_frame.jpg')

        # Uploader de frames em segundo plano para a API
        self.frame_queue = queue.Queue(maxsize=1)
        self.uploader_thread = threading.Thread(target=self._frame_uploader_loop, daemon=True)
        self.uploader_thread.start()

        # Configuração de zonas de prateleira
        zone_cfg = self.config.data.get('zone_tracking', {})
        self.zones = zone_cfg.get('zones', [])
        self.zone_tracking_enabled = bool(zone_cfg.get('enabled', False))
        self.draw_zones_enabled = bool(zone_cfg.get('draw_zones', True))
        self.track_hand_last_zone = {}

        # Configuração de detecção de pegada em zonas
        self.zone_grab_detection = bool(zone_cfg.get('grab_detection_enabled', True))
        self.zone_grab_min_frames = int(zone_cfg.get('min_hold_frames', 10))
        self.zone_grab_entry_speed_threshold = float(zone_cfg.get('entry_speed_threshold', 5.0))
        self.zone_grab_arm_flex_threshold = float(zone_cfg.get('arm_flex_threshold', 0.85))
        self.zone_grab_deceleration_threshold = float(zone_cfg.get('deceleration_threshold', 0.5))
        self.track_zone_hold_frames = {}
        self.track_hand_last_pos = {}
        self.track_body_last_pos = {}
        self.track_zone_entry_info = {}

        # Configuração de saída
        camera_cfg = config.camera()
        viz_cfg = self.config.get('visualization', {})
        self.output_width = int(camera_cfg.get('width', 640))
        self.output_height = int(camera_cfg.get('height', 480))
        self.output_size = (self.output_width, self.output_height)
        
        self.max_display_width = int(viz_cfg.get('max_display_width', 1280))
        self.max_display_height = int(viz_cfg.get('max_display_height', 850))

    @staticmethod
    def _resolve_node_id() -> str:
        env_id = os.getenv('NODE_ID', '').strip()
        if env_id:
            print(f"[NODE] ID configurado via env: {env_id}")
            return env_id
        try:
            name = socket.gethostname().strip().lower()
            safe = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")
            node_id = safe if safe.startswith("node") else f"node-{safe}"
            print(f"[NODE] ID derivado do hostname '{name}': {node_id}")
            return node_id
        except Exception:
            return "node-unknown"

    def _print_startup(self):
        print("\n" + "=" * 60)
        print("SISTEMA ANTI-FURTO - MODULAR PIPELINE")
        print("=" * 60)
        print(f"Backend: {self.detector_info.get('backend', 'N/A')}")
        print(f"Atividades carregadas: {len(self.activities)}")
        print(f"Handlers de alerta: {len(self.alert_handlers)}")
        temporal_status = "[OK] ATIVO" if self.temporal_filter.is_enabled() else "desativado"
        print(f"Filtro Temporal: {temporal_status}")
        norm_status = "[OK] ATIVO" if self.show_normalized_window else "desativado"
        print(f"Janela Normalizada: {norm_status}")
        print("Controles: Q = sair | D = debug | N = toggle normalizado | T = toggle temporal")
        print("=" * 60 + "\n")

    def _count_people(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return 0
        return len(keypoints) if np.asarray(keypoints).ndim == 3 else 1

    def _build_info_lines(self, had_new_inference: bool):
        return [
            f"FPS: {self.metrics.fps:.1f}",
            f"Frame: {self.metrics.frame_count}",
            f"People: {self._count_people(self.last_detection[0])}",
            f"Inference: {self.last_inference_ms:.1f}ms ({'NEW' if had_new_inference else 'CACHED'})",
            f"Uptime: {int(self.metrics.uptime_seconds())}s",
            f"Success: {self.metrics.success_rate():.0f}%",
        ]

    def _print_summary(self):
        print("\n" + "=" * 60)
        print("RESUMO DE EXECUCAO")
        print("=" * 60)
        print(f"Frames processados: {self.metrics.frame_count}")
        print(f"Inferencias executadas: {self.metrics.inference_calls}")
        print(f"Deteccoes com keypoints: {self.metrics.detection_count}")
        print(f"Tempo medio inferencia: {self.metrics.average_inference_ms():.2f}ms")
        print(f"Taxa de sucesso: {self.metrics.success_rate():.1f}%")
        print("=" * 60)
        for handler in self.alert_handlers:
            summary_fn = getattr(handler, "imprime_resumo", None)
            if callable(summary_fn):
                summary_fn()

    def _guardar_metricas(self):
        """Delega as métricas para o cofre local. O Dispatcher trata do resto."""
        try:
            metricas_data = {
                "node_id": self.node_id,
                "timestamp": time.time(),
                "fps": self.metrics.fps,
                "frame_count": self.metrics.frame_count,
                "detection_count": self.metrics.detection_count,
                "inference_calls": self.metrics.inference_calls,
                "average_inference_ms": self.metrics.average_inference_ms(),
                "success_rate": self.metrics.success_rate(),
                "uptime_seconds": self.metrics.uptime_seconds(),
                "pessoas_detetadas": self._count_people(self.last_detection[0])
            }
            self.db.salvar_metrica(metricas_data)
        except Exception as e:
            print(f"[ERRO] Falha ao enviar metrica para o cofre local: {str(e)}")

    def _guardar_frame_web(self, frame):
        """Guarda o último frame processado para stream web e coloca na fila de upload."""
        try:
            altura, largura = frame.shape[:2]
            if largura > 1280:
                escala = 1280 / largura
                frame_redimensionado = cv2.resize(frame, (1280, int(altura * escala)))
            else:
                frame_redimensionado = frame
            
            # Codifica em JPEG em memória
            success, encoded_image = cv2.imencode('.jpg', frame_redimensionado, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                jpeg_bytes = encoded_image.tobytes()
                # Tenta colocar na fila (se cheia, retira o anterior para manter sempre o mais fresco)
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(jpeg_bytes)

            api_url = self.runtime_config.get("api_url", "")

            # TODO: Repensar se vale a pena
            # Só grava em disco se não houver uploader ativo
            if not api_url:
                cv2.imwrite(self.frame_web_path, frame_redimensionado, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:
            print(f"[AVISO] Falha ao guardar frame web: {str(e)}")

    def _frame_uploader_loop(self):
        """Loop executado em segundo plano para fazer upload dos frames via POST HTTP"""
        api_url = self.runtime_config.get("api_url", "").strip()
        if not api_url:
            print("[UPLOADER] API URL não configurada. Upload de frames desativado.")
            return

        print(f"[UPLOADER] Iniciado uploader para o nó '{self.node_id}' -> {api_url}")
        upload_url = f"{api_url}/api/video/upload/{self.node_id}"

        while True:
            try:
                frame_data = self.frame_queue.get()
                if frame_data is None:
                    break

                try:
                    # Envia como corpo binário (raw data)
                    requests.post(
                        upload_url, 
                        data=frame_data, 
                        headers={"Content-Type": "image/jpeg"}, 
                        timeout=2.0
                    )
                except Exception:
                    # Falhas de rede são silenciosas para não inundar a consola
                    pass
            except Exception as e:
                time.sleep(1)

    @staticmethod
    def _box_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _build_bbox(self, keypoints, scores, frame_shape, padding):
        h_img, w_img = frame_shape[:2]
        in_bounds = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w_img) & \
                    (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h_img) & \
                    np.isfinite(keypoints).all(axis=1)
        
        valid_mask = in_bounds & (scores > 0.2)
        if np.sum(valid_mask) < 3:
            valid_mask = in_bounds

        if not np.any(valid_mask):
            return None

        valid_kpts = keypoints[valid_mask]
        x_min, y_min = np.min(valid_kpts, axis=0)
        x_max, y_max = np.max(valid_kpts, axis=0)

        pad_x = max(float(padding.get('x', 25)), (x_max - x_min) * 0.12)
        pad_y = max(float(padding.get('y', 35)), (y_max - y_min) * 0.18)

        x1 = int(np.clip(x_min - pad_x, 0, w_img - 1))
        y1 = int(np.clip(y_min - pad_y, 0, h_img - 1))
        x2 = int(np.clip(x_max + pad_x, 0, w_img - 1))
        y2 = int(np.clip(y_max + pad_y, 0, h_img - 1))

        return [x1, y1, x2, y2, float(np.mean(scores[valid_mask])), 0.0] if x2 > x1 and y2 > y1 else None

    def _scale_zone_rect(self, rect):
        """Escala as coordenadas da zona do config (assumindo base 640x480) para a resolucao real do frame."""
        x1, y1, x2, y2 = rect
        base_w = float(self.config.camera().get('width', 640))
        base_h = float(self.config.camera().get('height', 480))
        
        cur_w, cur_h = getattr(self, 'current_frame_size', (base_w, base_h))
        
        sx = cur_w / base_w
        sy = cur_h / base_h
        return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

    def _draw_zones(self, frame):
        if not self.draw_zones_enabled or not self.zone_tracking_enabled or not self.zones:
            return

        overlay = frame.copy()
        for zone in self.zones:
            rect = zone.get('rect')
            if not rect or len(rect) != 4:
                continue
            x1, y1, x2, y2 = self._scale_zone_rect(rect)
            color = tuple(int(c) for c in zone.get('color', [0, 255, 255]))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.putText(overlay, zone.get('name', f"Zona {zone.get('id', '?')}"),
                        (x1 + 8, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        alpha = 0.10
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        for zone in self.zones:
            rect = zone.get('rect')
            if not rect or len(rect) != 4:
                continue
            x1, y1, x2, y2 = self._scale_zone_rect(rect)
            border_color = tuple(min(255, int(c * 1.2)) for c in zone.get('color', [0, 255, 255]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    def _get_zone_for_point(self, x, y):
        for zone in self.zones:
            rect = zone.get('rect')
            if not rect or len(rect) != 4:
                continue
            x1, y1, x2, y2 = self._scale_zone_rect(rect)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone
        return None

    def _compute_body_center(self, kpts, scrs):
        candidates = []
        if scrs.shape[0] > LEFT_HIP and scrs[LEFT_HIP] >= 0.35:
            candidates.append(kpts[LEFT_HIP])
        if scrs.shape[0] > RIGHT_HIP and scrs[RIGHT_HIP] >= 0.35:
            candidates.append(kpts[RIGHT_HIP])
        if not candidates:
            return None
        return np.mean(candidates, axis=0)

    def _compute_arm_length(self, kpts, scrs, elbow_idx, wrist_idx):
        """Calcula o comprimento do braço (elbow-to-wrist distance)."""
        if scrs.shape[0] <= max(elbow_idx, wrist_idx):
            return None
        if scrs[elbow_idx] < 0.35 or scrs[wrist_idx] < 0.35:
            return None
        elbow_pos = kpts[elbow_idx]
        wrist_pos = kpts[wrist_idx]
        return np.linalg.norm(wrist_pos - elbow_pos)

    def _check_hand_zone(self, ent, track_id, timestamp):
        kpts = np.asarray(ent['kpts'], dtype=np.float32)
        scrs = np.asarray(ent['scrs'], dtype=np.float32)
        if kpts.size == 0 or scrs.size == 0:
            return

        hands = []
        if scrs.shape[0] > LEFT_WRIST and scrs[LEFT_WRIST] >= 0.35:
            hands.append(('left', tuple(kpts[LEFT_WRIST].tolist())))
        if scrs.shape[0] > RIGHT_WRIST and scrs[RIGHT_WRIST] >= 0.35:
            hands.append(('right', tuple(kpts[RIGHT_WRIST].tolist())))

        body_center = self._compute_body_center(kpts, scrs)
        body_prev = self.track_body_last_pos.get(track_id)
        body_speed = 0.0
        if body_center is not None:
            if body_prev is not None:
                body_speed = np.linalg.norm(body_center - body_prev)
            self.track_body_last_pos[track_id] = body_center

        for hand_name, (x, y) in hands:
            elbow_idx = LEFT_ELBOW if hand_name == 'left' else RIGHT_ELBOW
            arm_length = self._compute_arm_length(kpts, scrs, elbow_idx, LEFT_WRIST if hand_name == 'left' else RIGHT_WRIST)
            
            zone = self._get_zone_for_point(x, y)
            zone_id = zone['id'] if zone else None
            previous_zone = self.track_hand_last_zone.get((track_id, hand_name))

            entry_key = (track_id, hand_name, zone_id)
            
            if zone_id != previous_zone:
                self.track_hand_last_zone[(track_id, hand_name)] = zone_id
                self.track_zone_hold_frames.pop((track_id, hand_name, previous_zone), None)
                self.track_zone_entry_info.pop(entry_key, None)
                self.track_hand_last_pos[(track_id, hand_name)] = np.array([x, y], dtype=np.float32)

            if zone:
                current_pos = np.array([x, y], dtype=np.float32)
                prev_pos = self.track_hand_last_pos.get((track_id, hand_name))
                speed = np.linalg.norm(current_pos - prev_pos) if prev_pos is not None else 0.0
                self.track_hand_last_pos[(track_id, hand_name)] = current_pos

                # Ao entrar na zona, guardar informações de entrada
                if entry_key not in self.track_zone_entry_info:
                    self.track_zone_entry_info[entry_key] = {
                        'entry_speed': speed,
                        'entry_arm_length': arm_length,
                        'frames_in_zone': 0
                    }

                entry_info = self.track_zone_entry_info[entry_key]
                entry_info['frames_in_zone'] += 1
                
                # Detectar grab: desaceleração + flexão do braço
                grab_criterion = False
                deceleration_ratio = 0.0
                arm_flex_ratio = 0.0
                
                if entry_info['entry_speed'] > self.zone_grab_entry_speed_threshold:
                    deceleration_ratio = speed / max(entry_info['entry_speed'], 0.1)
                    if deceleration_ratio <= self.zone_grab_deceleration_threshold:
                        grab_criterion = True
                
                if arm_length is not None and entry_info['entry_arm_length'] is not None:
                    arm_flex_ratio = arm_length / max(entry_info['entry_arm_length'], 1.0)
                    if arm_flex_ratio <= self.zone_grab_arm_flex_threshold and grab_criterion:
                        grab_criterion = True
                    elif arm_flex_ratio <= (self.zone_grab_arm_flex_threshold * 0.9):  # Flexão forte
                        grab_criterion = True
                
                hold_key = (track_id, hand_name, zone_id)
                if grab_criterion or speed < 1.5:  # Mão muito lenta também indica grab
                    self.track_zone_hold_frames[hold_key] = self.track_zone_hold_frames.get(hold_key, 0) + 1
                else:
                    self.track_zone_hold_frames[hold_key] = 0

                hold_frames = self.track_zone_hold_frames[hold_key]
                grab_detected = (self.zone_grab_detection and
                                 hold_frames >= self.zone_grab_min_frames)
                if grab_detected and hold_frames == self.zone_grab_min_frames:
                    print(f"[GRAB] ID {track_id} ({hand_name}) PEGOU ALGO em {zone['name']} (decel={deceleration_ratio:.2f}, flex={arm_flex_ratio:.2f})")
                    self.db.salvar_evento_zona(
                        track_id=int(track_id),
                        zone_id=int(zone_id),
                        zone_name=str(zone['name']),
                        hand=str(hand_name),
                        deceleration_ratio=float(deceleration_ratio),
                        arm_flex_ratio=float(arm_flex_ratio),
                        arm_length=float(arm_length),
                        timestamp=timestamp
                    )

                ent.setdefault('hand_zones', []).append({
                    'hand': hand_name,
                    'zone_id': zone_id,
                    'zone_name': zone['name'],
                    'point': (x, y),
                    'zone_hold_frames': int(hold_frames),
                    'grab_candidate': grab_detected,
                    'arm_length': float(arm_length) if arm_length else None,
                    'deceleration_ratio': float(deceleration_ratio),
                    'arm_flex_ratio': float(arm_flex_ratio),
                })

    def _processar_zonas(self, entidades, timestamp):
        if not self.zone_tracking_enabled or not self.zones:
            return

        for ent in entidades:
            track_id = ent.get('id')
            if track_id is None or track_id == '...':
                continue
            self._check_hand_zone(ent, track_id, timestamp)

    def _deduplicate_entities(self, entidades, iou_threshold=0.85):
        if len(entidades) < 2:
            return entidades
        entidades_ordenadas = sorted(entidades, key=lambda ent: ent['box'][4], reverse=True)
        filtradas = []
        for entidade in entidades_ordenadas:
            is_duplicate = any(
                self._box_iou(entidade['box'], kept['box']) >= iou_threshold
                for kept in filtradas
            )
            if not is_duplicate:
                filtradas.append(entidade)
        return filtradas

    def _preparar_entidades(self, keypoints, scores, frame_shape):
        """Calcula bboxes para 1 ou N pessoas."""
        if keypoints is None or scores is None:
            return []

        kpts_arr = np.asarray(keypoints, dtype=np.float32)
        scrs_arr = np.asarray(scores, dtype=np.float32)

        if kpts_arr.size == 0 or scrs_arr.size == 0:
            return []

        if kpts_arr.ndim == 2:
            kpts_arr = kpts_arr[np.newaxis, :, :]
        if scrs_arr.ndim == 1:
            scrs_arr = scrs_arr[np.newaxis, :]

        padding = self.config.visualization().get('bbox_padding', {'x': 25, 'y': 35})
        entidades = []
        for scrs, kpts in zip(scrs_arr, kpts_arr):
            bbox = self._build_bbox(kpts, scrs, frame_shape, padding)
            if bbox is None:
                continue
            entidades.append({'kpts': kpts.tolist(), 'scrs': scrs.tolist(), 'box': bbox, 'id': '...'})
        tracker_cfg = self.config.tracker()
        dedup_thresh = float(tracker_cfg.get("dedup_thresh", 0.55))
        return self._deduplicate_entities(entidades, iou_threshold=dedup_thresh)

    def _atribuir_ids(self, entidades):
        """Atribui IDs do ByteTrack às entidades usando IoU com as caixas rastreadas."""
        if not entidades or self.last_tracked_objects is None or len(self.last_tracked_objects) == 0:
            return

        remaining = [
            {'box': t[0:4].tolist(), 'id': int(t[4])}
            for t in np.asarray(self.last_tracked_objects) if len(t) >= 5
        ]

        for ent in entidades:
            best_iou, best_track = 0.15, None
            for track in remaining:
                iou = self._box_iou(ent['box'], track['box'])
                if iou >= best_iou:
                    best_iou, best_track = iou, track
            if best_track:
                ent['id'] = best_track['id']
                remaining.remove(best_track)

    def _processar_atividades(self, entidades, timestamp):
        """Executa a lógica de detecção de furto e grava na base de dados SQLite."""
        alert_text = None
        for ent in entidades:
            track_id = ent.get('id')
            if track_id is None or track_id == '...':
                continue

            # Normaliza a pose uma única vez por entidade e guarda para reutilização
            kpts_arr = np.asarray(ent['kpts'], dtype=np.float32)
            scrs_arr = np.asarray(ent['scrs'], dtype=np.float32)
            norm_pose = self.normalizer.normalize(kpts_arr, scrs_arr)
            ent['norm_pose'] = norm_pose

            for activity in self.activities:
                event = activity.detecta(norm_pose, self.metrics.frame_count, timestamp, track_id=track_id)
                if event:
                    for handler in self.alert_handlers:
                        handler.registra_evento(event)
                    alert_text = f"ALERTA: {event.tipo} ({event.confianca:.0%})"
                    self.db.salvar_alerta(track_id, event.tipo, event.confianca * 100)
                    print(f"\n>>> [ALERTA DETETADO] ID {track_id}: {event.tipo.upper()} ({event.confianca:.0%}) - {event.descricao}\n")
        return alert_text

    def _desenhar_caixas(self, frame, entidades):
        """Desenha bboxes e IDs no frame final."""
        for ent in entidades:
            x1, y1, x2, y2, _, _ = ent['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {ent['id']}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _processar_frame(self, keypoints, scores, frame_shape, timestamp):
        """Processa poses, tracking, filtro temporal e atividades."""
        if keypoints is None or scores is None:
            return [], None

        entidades = self._preparar_entidades(keypoints, scores, frame_shape)
        if not entidades:
            self.last_tracked_objects = []
            return [], None

        bboxes = TorchTensorMock([ent['box'] for ent in entidades])
        self.last_tracked_objects = self.tracker.update(bboxes, frame_shape[:2])
        self.metrics.on_detection()
        self._atribuir_ids(entidades)

        # Limpa buffers de tracks que já não estão presentes
        ids_presentes = {ent['id'] for ent in entidades if ent.get('id') not in (None, '...')}
        for activity in self.activities:
            activity.limpa_tracks_inativas(ids_presentes)

        if self.temporal_filter.is_enabled():
            # Prune filtros de tracks desaparecidas
            self.temporal_filters_por_id = {
                tid: f for tid, f in self.temporal_filters_por_id.items()
                if tid in ids_presentes
            }

            for ent in entidades:
                track_id = ent.get('id')
                if track_id is not None:
                    filter_obj = self.temporal_filters_por_id.setdefault(track_id, TemporalPoseFilter(self.config))
                kpts_arr = np.asarray(ent['kpts'], dtype=np.float32)
                scrs_arr = np.asarray(ent['scrs'], dtype=np.float32)
                filt_kpts, filt_scrs, _ = filter_obj.filter_pose(kpts_arr, scrs_arr)
                ent['kpts'] = filt_kpts.tolist()
                ent['scrs'] = filt_scrs.tolist()

        self._processar_zonas(entidades, timestamp)
        alert_text = self._processar_atividades(entidades, timestamp)
        return entidades, alert_text

    def _render_normalized_canvas(self, entidades):
        """Renderiza a janela de debug de poses normalizadas (apenas se show_normalized_window=True)."""
        if not entidades or not self.show_normalized_window or self.skeleton_viz is None:
            return None

        poses = []
        for ent in entidades[:4]:
            # Reutiliza a norm_pose já calculada em _processar_atividades (sem re-normalizar)
            norm_pose = ent.get('norm_pose')
            if norm_pose is None:
                continue

            if norm_pose.is_valid:
                title = f"ID {ent.get('id', '?')} | Torso: {norm_pose.torso_length:.1f}px"
                canvas = self.skeleton_viz.render(norm_pose.keypoints, norm_pose.scores, title=title)
            else:
                canvas = np.zeros((self.skeleton_viz.canvas_size, self.skeleton_viz.canvas_size, 3), dtype=np.uint8)
                cv2.putText(canvas, f"ID {ent.get('id', '?')} - Frame invalido",
                            (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            poses.append(canvas)

        if not poses:
            return None

        cell_size = self.skeleton_viz.canvas_size // 2
        grid = np.zeros((self.skeleton_viz.canvas_size, self.skeleton_viz.canvas_size, 3), dtype=np.uint8)
        for idx, canvas in enumerate(poses[:4]):
            y1, x1 = (idx // 2) * cell_size, (idx % 2) * cell_size
            grid[y1:y1 + cell_size, x1:x1 + cell_size] = cv2.resize(canvas, (cell_size, cell_size))
        return grid

    def _resize_to_fit(self, img):
        """Redimensiona uma imagem mantendo o aspect ratio para caber perfeitamente no ecrã."""
        if img is None:
            return None
        orig_h, orig_w = img.shape[:2]
        
        # Obter resolução máxima de visualização (fallback seguro)
        max_w = getattr(self, 'max_display_width', 1280)
        max_h = getattr(self, 'max_display_height', 850)
        
        escala = min(max_w / orig_w, max_h / orig_h)
        nova_largura = int(orig_w * escala)
        nova_altura = int(orig_h * escala)
        
        # Usa interpolacao correta dependendo se estamos a aumentar ou diminuir a imagem
        interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(img, (nova_largura, nova_altura), interpolation=interp)

    def run(self):
        self._print_startup()
        cap = self.video_source.open()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.current_frame_size = (frame.shape[1], frame.shape[0])

                if self.metrics.frame_count == 0:
                    print(f"[VIDEO] Resolucao Real: {frame.shape[1]}x{frame.shape[0]}")


                self.metrics.on_frame()
                timestamp = time.time()

                # Guardar métricas periodicamente
                if (self.metrics.frame_count - self.ultimo_frame_metricas) >= self.metricas_intervalo:
                    self._guardar_metricas()
                    self.ultimo_frame_metricas = self.metrics.frame_count

                should_infer = (self.metrics.frame_count % self.metrics.frame_skip) == 0
                if should_infer:
                    t0 = time.time()
                    keypoints, scores = self.detector.detect(frame)
                    self.last_inference_ms = (time.time() - t0) * 1000.0
                    self.metrics.on_inference(self.last_inference_ms)
                    self.last_detection = (keypoints, scores)
                elif not self.cache_result:
                    self.last_detection = ([], [])

                keypoints, scores = self.last_detection

                entidades, new_alert_text = [], None
                if keypoints is not None and len(keypoints) > 0:
                    entidades, new_alert_text = self._processar_frame(keypoints, scores, frame.shape, timestamp)

                # Persistência do texto do alerta no ecrã
                if new_alert_text:
                    self.current_alert_text = new_alert_text
                    self.current_alert_countdown = self.alert_persist_frames
                elif self.current_alert_countdown > 0:
                    self.current_alert_countdown -= 1
                    if self.current_alert_countdown == 0:
                        self.current_alert_text = None

                # Render: usa entidades processadas para alinhar skeleton com caixas
                render_kpts = [ent['kpts'] for ent in entidades] if entidades else keypoints
                render_scrs = [ent['scrs'] for ent in entidades] if entidades else scores

                output = self.renderer.render(frame, render_kpts, render_scrs)
                self._draw_zones(output)
                self._desenhar_caixas(output, entidades)
                self.renderer.draw_overlay(output, self._build_info_lines(should_infer),
                                           alert_text=self.current_alert_text, debug=self.debug)

                # Frame web
                if (self.metrics.frame_count - self.ultimo_frame_web) >= self.frame_web_intervalo:
                    self._guardar_frame_web(output)
                    self.ultimo_frame_web = self.metrics.frame_count

                # Redimensionar e exibir janelas mantendo aspect ratio
                cv2.imshow(self.renderer.window_name, self._resize_to_fit(output))

                sk_canvas = getattr(self.renderer, 'last_canvas', None)
                if sk_canvas is not None:
                    try:
                        cv2.imshow(self.renderer.window_name + ' - SKELETON', self._resize_to_fit(sk_canvas))
                    except Exception:
                        pass

                if self.show_normalized_window:
                    normalized_canvas = self._render_normalized_canvas(entidades)
                    if normalized_canvas is not None:
                        try:
                            cv2.imshow(self.renderer.window_name + ' - NORMALIZED', self._resize_to_fit(normalized_canvas))
                        except Exception:
                            pass

                # Controlos de teclado
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("d"), ord("D")):
                    self.debug = not self.debug
                    print(f"[DEBUG] {'ATIVADO' if self.debug else 'DESATIVADO'}")
                if key in (ord("n"), ord("N")) and self.show_normalized_window:
                    self.show_normalized_window = not self.show_normalized_window
                    print(f"[NORMALIZADO] {'VISIVEL' if self.show_normalized_window else 'OCULTO'}")
                if key in (ord("t"), ord("T")):
                    self.temporal_filter.toggle()
                    if not self.temporal_filter.is_enabled():
                        self.temporal_filters_por_id.clear()
                    print(f"[TEMPORAL] Filtro {'ATIVO' if self.temporal_filter.is_enabled() else 'INATIVO'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_summary()
            
            # Antes de fechar, garante que as métricas finais são salvas
            self.db.close()