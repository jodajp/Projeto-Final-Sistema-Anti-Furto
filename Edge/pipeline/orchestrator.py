import time
import cv2
import numpy as np
import os
import socket
from datetime import datetime
from pathlib import Path

from Alertas.database_handler import DatabaseHandler
from bytetracker import BYTETracker
from Edge.Detecao.detector_factory import create_detector
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
        self.frame_web_intervalo = 10
        self.ultimo_frame_web = 0
        self.frame_web_path = os.path.join(self.metricas_dir, 'last_frame.jpg')

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
        """Guarda o último frame processado para stream web."""
        try:
            altura, largura = frame.shape[:2]
            if largura > 1280:
                escala = 1280 / largura
                frame_redimensionado = cv2.resize(frame, (1280, int(altura * escala)))
            else:
                frame_redimensionado = frame
            cv2.imwrite(self.frame_web_path, frame_redimensionado, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:
            print(f"[AVISO] Falha ao guardar frame web: {str(e)}")

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
        return self._deduplicate_entities(entidades)

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

    def run(self):
        self._print_startup()
        cap = self.video_source.open()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

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
                self._desenhar_caixas(output, entidades)
                self.renderer.draw_overlay(output, self._build_info_lines(should_infer),
                                           alert_text=self.current_alert_text, debug=self.debug)

                # Frame web
                if (self.metrics.frame_count - self.ultimo_frame_web) >= self.frame_web_intervalo:
                    self._guardar_frame_web(output)
                    self.ultimo_frame_web = self.metrics.frame_count

                cv2.imshow(self.renderer.window_name, output)

                sk_canvas = getattr(self.renderer, 'last_canvas', None)
                if sk_canvas is not None:
                    try:
                        cv2.imshow(self.renderer.window_name + ' - SKELETON', sk_canvas)
                    except Exception:
                        pass

                if self.show_normalized_window:
                    normalized_canvas = self._render_normalized_canvas(entidades)
                    if normalized_canvas is not None:
                        try:
                            cv2.imshow(self.renderer.window_name + ' - NORMALIZED', normalized_canvas)
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