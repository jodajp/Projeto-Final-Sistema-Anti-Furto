import os
import time
import json
import cv2
import numpy as np
import torch

from .spatial_normalizer import SpatialNormalizer, NormalizationParams
from .skeleton_visualizer import SkeletonVisualizer
from .temporal_pose_filter import TemporalPoseFilter, TemporalPoseFilterConfig
from Alertas.database_handler import DatabaseHandler
from bytetracker import BYTETracker
from Detecao.detector_factory import create_detector
from .activity_loader import load_activities
from .alert_dispatcher import AlertDispatcher, load_alert_handlers
from .metrics import PipelineMetrics
from .renderer import PoseRenderer
from .video_source import VideoSource


class AntiTheftOrchestrator:
    """Coordena detector, atividades, alertas, render e loop de video."""

    def __init__(self, config):
        self.config = config
        self.runtime_config = config.runtime()

        self.detector = create_detector(config.detector_config())
        self.detector_info = self.detector.get_info() or {}

        self.activities = load_activities(config.activity_specs())
        handlers = load_alert_handlers(config.alert_specs())
        self.alert_dispatcher = AlertDispatcher(handlers)

        self.renderer = PoseRenderer(config.visualization())
        self.video_source = VideoSource(config.camera())
        self.metrics = PipelineMetrics(frame_skip=config.frame_skip())

        self.cache_result = bool(self.runtime_config.get("cache_result", True))
        self.debug = bool(self.runtime_config.get("debug", False))

        self.last_detection = ([], [])
        self.last_inference_ms = 0.0

        self.tracker = BYTETracker(
            track_thresh=0.5, 
            track_buffer=30, 
            match_thresh=0.8
        )
        self.last_tracked_objects = []

        self.db = DatabaseHandler()
        
        # Initialize temporal pose filtering
        temporal_cfg = config.temporal_filter_config()
        if temporal_cfg.get("enabled", True):
            self.temporal_filter_enabled = True
            temporal_filter_config = TemporalPoseFilterConfig(
                enabled=temporal_cfg.get("enabled", True),
                smoothing_factor=temporal_cfg.get("smoothing_factor", 0.6),
                smoothing_factor_fast=temporal_cfg.get("smoothing_factor_fast", 0.85),
                rapid_movement_threshold=temporal_cfg.get("rapid_movement_threshold", 5.0),
                velocity_smoothing=temporal_cfg.get("velocity_smoothing", 0.3),
                occlusion_confidence_threshold=temporal_cfg.get("occlusion_confidence_threshold", 0.3),
                max_occlusion_frames=temporal_cfg.get("max_occlusion_frames", 5),
                velocity_damping=temporal_cfg.get("velocity_damping", 0.94),
            )
            self.temporal_filter = TemporalPoseFilter(temporal_filter_config)
        else:
            self.temporal_filter_enabled = False
            self.temporal_filter = None

        # Initialize spatial normalization
        spatial_cfg = config.data.get("spatial_normalization", {})
        if spatial_cfg.get("enabled", False):
            self.spatial_norm_enabled = True
            norm_params = NormalizationParams(
                torso_confidence_threshold=spatial_cfg.get("torso_confidence_threshold", 0.5),
                min_torso_length_px=spatial_cfg.get("min_torso_length_px", 10.0),
                allow_invalid_torso=spatial_cfg.get("allow_invalid_torso", False),
            )
            self.normalizer = SpatialNormalizer(norm_params)
            self.skeleton_viz = SkeletonVisualizer()
            self.show_normalized = True
        else:
            self.spatial_norm_enabled = False
            self.normalizer = None
            self.skeleton_viz = None
            self.show_normalized = False

        self.last_detection = ([], [])
        self.last_inference_ms = 0.0
        self.temporal_filter_applied = False


    def _print_startup(self):
        print("\n" + "=" * 60)
        print("SISTEMA ANTI-FURTO - MODULAR PIPELINE")
        print("=" * 60)
        print(f"Backend: {self.detector_info.get('backend', 'N/A')}")
        print(f"Atividades carregadas: {len(self.activities)}")
        print(f"Handlers de alerta: {len(self.alert_dispatcher.handlers)}")
        print("Controles: Q = sair | D = toggle debug")
        print("=" * 60 + "\n")

    def _build_info_lines(self, had_new_inference: bool):
        keypoints_ref = self.last_detection[0]
        keypoint_count = len(keypoints_ref) if keypoints_ref is not None else 0
        inference_state = "NEW" if had_new_inference else "CACHED"

        return [
            f"FPS: {self.metrics.fps:.1f}",
            f"Frame: {self.metrics.frame_count}",
            f"Keypoints: {keypoint_count}",
            f"Inference: {self.last_inference_ms:.1f}ms ({inference_state})",
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
        self.alert_dispatcher.print_summary()

    def _preparar_entidades(self, keypoints, scores, frame_shape):
        """Calcula bboxes e normaliza dados para 0, 1 ou N pessoas sem erros de ambiguidade."""
        if keypoints is None or scores is None or len(keypoints) == 0:
            return []
        
        # ndmin trata a escala automaticamente (0, 1 ou N pessoas)
        kpts_arr = np.array(keypoints, ndmin=3)
        scrs_arr = np.array(scores, ndmin=2)
        h_img, w_img = frame_shape[:2]

        # DETECÇÃO DE ESCALA: Em vez de /2 fixo, calculamos o rácio real.
        # Se os pontos vêm em 1280 e a imagem é 640, o scale será 0.5 (equivalente ao /2)
        # Se a imagem for igual ao modelo, o scale será 1.0 (não estraga nada)
        input_w = self.detector_info.get('input_width', w_img * 2 if "/2" else w_img) 
        scale_x = w_img / input_w
        scale_y = h_img / self.detector_info.get('input_height', h_img * 2 if "/2" else h_img)

        viz_cfg = self.config.visualization()
        padding = viz_cfg.get('bbox_padding', {'x': 25, 'y': 35})
        conf_min = viz_cfg.get('confidence_threshold', 0.3)

        entidades = []
        for scrs, kpts in zip(scrs_arr, kpts_arr):
            mask = scrs > conf_min
            if not np.any(mask): continue 

            # Aplica a escala correctiva (O substituto profissional do /2)
            kpts_scaled = kpts.copy()
            kpts_scaled[:, 0] *= scale_x
            kpts_scaled[:, 1] *= scale_y

            kpts_vis = kpts_scaled[mask]
            x_min, y_min = np.min(kpts_vis, axis=0)
            x_max, y_max = np.max(kpts_vis, axis=0)

            x1 = int(np.clip(x_min - padding['x'], 0, w_img))
            y1 = int(np.clip(y_min - padding['y'], 0, h_img))
            x2 = int(np.clip(x_max + padding['x'], 0, w_img))
            y2 = int(np.clip(y_max + padding['y'], 0, h_img))

            entidades.append({
                'kpts': kpts_scaled.tolist(),
                'scrs': scrs.tolist(),
                'box': [x1, y1, x2, y2, float(np.mean(scrs[mask])), 0.0],
                'center_x': (x1 + x2) / 2,
                'id': "..."
            })
        return entidades

    def _atribuir_ids(self, entidades):
        """Cruza os IDs do ByteTrack com as bboxes originais por posição horizontal."""
        if not entidades or self.last_tracked_objects is None or len(self.last_tracked_objects) == 0:
            return

        objetos_tracker = []
        for obj in self.last_tracked_objects:
            if hasattr(obj, 'track_id'):
                track_id = obj.track_id
                tx1, tx2 = float(obj.tlbr[0]), float(obj.tlbr[2])
            else:
                tx1, tx2, track_id = float(obj[0]), float(obj[2]), int(obj[4])
            objetos_tracker.append({'id': track_id, 'center_x': (tx1 + tx2) / 2})

        entidades.sort(key=lambda e: e['center_x'])
        objetos_tracker.sort(key=lambda t: t['center_x'])

        for i, entidade in enumerate(entidades):
            if i < len(objetos_tracker):
                entidade['id'] = objetos_tracker[i]['id']

    def _processar_atividades(self, entidades, timestamp):
        """Executa a lógica de detecção de furto e grava na base de dados SQLite."""
        alert_text = None
        for ent in entidades:
            for activity in self.activities:
                event = activity.detecta(ent['kpts'], ent['scrs'], self.metrics.frame_count, timestamp)
                if event:
                    self.alert_dispatcher.dispatch(event)
                    alert_text = f"ALERTA: {event.tipo} ({event.confianca:.0%})"
                    self.db.salvar_alerta(ent['id'], event.tipo, event.confianca * 100)
        return alert_text

    def _desenhar_caixas(self, frame, entidades):
        """Desenha bboxes e IDs no frame final."""
        for ent in entidades:
            x1, y1, x2, y2, _, _ = ent['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {ent['id']}", (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        self._print_startup()
        cap = self.video_source.open()

        try:
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                self.metrics.on_frame()
                timestamp = time.time()

                should_infer = (self.metrics.frame_count % self.metrics.frame_skip) == 0
                if should_infer:
                    keypoints, scores = self.detector.detect(frame)
                    self.last_detection = (keypoints, scores)
                else:
                    keypoints, scores = self.last_detection

                if self.temporal_filter_enabled and len(keypoints) > 0:
                    k_arr, s_arr = np.array(keypoints), np.array(scores)
                    if k_arr.shape == (17, 2):
                        keypoints, scores, _ = self.temporal_filter.filter_pose(k_arr, s_arr)
                        keypoints, scores = keypoints.tolist(), scores.tolist()

                entidades = self._preparar_entidades(keypoints, scores, frame.shape)
                if should_infer and entidades:
                    bboxes = torch.tensor([e['box'] for e in entidades], dtype=torch.float32)
                    self.last_tracked_objects = self.tracker.update(bboxes, frame.shape[:2])
                
                self._atribuir_ids(entidades)
                alert_text = self._processar_atividades(entidades, timestamp)

                # --- 5. RENDERIZAÇÃO E INTERFACE ---
                # Criamos a imagem com esqueletos
                output = self.renderer.render(frame, keypoints, scores)
                
                # Desenhamos as nossas caixas multi-pessoa recicladas
                self._desenhar_caixas(output, entidades)

                # Mostramos o Overlay de Debug (FPS, Deteções, etc.)
                info = self._build_info_lines(should_infer)
                self.renderer.draw_overlay(output, info, alert_text, self.debug)

                # GARANTIA VISUAL: Forçamos a exibição da janela
                win_name = self.renderer.window_name or "SISTEMA ANTI-FURTO"
                cv2.imshow(win_name, output)

                if self.spatial_norm_enabled and self.show_normalized and len(keypoints) > 0:
                    self._render_esqueleto_normalizado(keypoints, scores)

                # --- COMANDOS DE TECLADO ---
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("d"), ord("D")):
                    self.debug = not self.debug
                    state = "ATIVADO" if self.debug else "DESATIVADO"
                    print(f"[DEBUG] {state}")
                if key in (ord("n"), ord("N")) and self.spatial_norm_enabled:
                    self.show_normalized = not self.show_normalized
                    state = "VISIVEL" if self.show_normalized else "OCULTO"
                    print(f"[NORMALIZADO] {state}")
                if key in (ord("t"), ord("T")) and self.temporal_filter_enabled:
                    # Toggle temporal filter on/off
                    self.temporal_filter_enabled = not self.temporal_filter_enabled
                    state = "ATIVO" if self.temporal_filter_enabled else "INATIVO"
                    print(f"[TEMPORAL] Filtro {state}")
                    if not self.temporal_filter_enabled:
                        self.temporal_filter.reset()
        finally:
            cap.release()
            cv2.destroyAllWindows()