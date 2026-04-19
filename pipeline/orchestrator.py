import time

import cv2
import numpy as np

from Detecao.detector_factory import create_detector
from .activity_loader import load_activities
from .alert_dispatcher import AlertDispatcher, load_alert_handlers
from .metrics import PipelineMetrics
from .renderer import PoseRenderer
from .video_source import VideoSource
from .spatial_normalizer import SpatialNormalizer, NormalizationParams
from .skeleton_visualizer import SkeletonVisualizer
from .temporal_pose_filter import TemporalPoseFilter, TemporalPoseFilterConfig


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
        temporal_status = "✓ ATIVO" if self.temporal_filter_enabled else "desativado"
        print(f"Filtro Temporal: {temporal_status}")
        spatial_status = "✓ ATIVO" if self.spatial_norm_enabled else "desativado"
        print(f"Normalização Espacial: {spatial_status}")
        print("Controles: Q = sair | D = debug | N = toggle normalizado | T = toggle temporal")
        print("=" * 60 + "\n")

    def _build_info_lines(self, had_new_inference: bool):
        keypoint_count = len(self.last_detection[0]) if self.last_detection else 0
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

    def run(self):
        self._print_startup()
        cap = self.video_source.open()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.metrics.on_frame()
                timestamp = time.time()

                should_infer = (self.metrics.frame_count % self.metrics.frame_skip) == 0
                if should_infer:
                    t0 = time.time()
                    detection = self.detector.detect(frame)
                    keypoints, scores = [], []
                    if isinstance(detection, tuple) and len(detection) == 2:
                        if detection[0] is not None:
                            keypoints = list(detection[0])
                        if detection[1] is not None:
                            scores = list(detection[1])
                    self.last_inference_ms = (time.time() - t0) * 1000.0

                    self.metrics.on_inference(self.last_inference_ms)
                    self.last_detection = (keypoints, scores)

                    if keypoints:
                        self.metrics.on_detection()
                elif not self.cache_result:
                    self.last_detection = ([], [])

                keypoints, scores = self.last_detection

                # Apply temporal filtering for jitter reduction and occlusion prediction
                if self.temporal_filter_enabled and keypoints:
                    try:
                        keypoints_array = np.array(keypoints, dtype=np.float32)
                        scores_array = np.array(scores, dtype=np.float32)
                        
                        # Ensure proper shape
                        if keypoints_array.shape == (17, 2) and scores_array.shape == (17,):
                            filtered_kpts, filtered_scores, was_predicted = self.temporal_filter.filter_pose(
                                keypoints_array, scores_array
                            )
                            keypoints = filtered_kpts.tolist()
                            scores = filtered_scores.tolist()
                            self.temporal_filter_applied = True
                    except Exception as e:
                        if self.debug:
                            print(f"[TEMPORAL] Erro ao filtrar: {e}")

                alert_text = None
                if keypoints:
                    for activity in self.activities:
                        event = activity.detecta(
                            keypoints,
                            scores,
                            self.metrics.frame_count,
                            timestamp,
                        )
                        if event:
                            self.alert_dispatcher.dispatch(event)
                            alert_text = f"ALERTA: {event.tipo} ({event.confianca:.0%})"

                output = self.renderer.render(frame, keypoints, scores)
                info_lines = self._build_info_lines(had_new_inference=should_infer)
                self.renderer.draw_overlay(
                    output,
                    info_lines,
                    alert_text=alert_text,
                    debug=self.debug,
                )

                cv2.imshow(self.renderer.window_name, output)

                # Render normalized skeleton if enabled
                if self.spatial_norm_enabled and self.show_normalized and keypoints:
                    try:
                        keypoints_array = np.array(keypoints, dtype=np.float32)
                        scores_array = np.array(scores, dtype=np.float32)
                        
                        normalized = self.normalizer.normalize(keypoints_array, scores_array)
                        
                        if normalized.is_valid:
                            normalized_canvas = self.skeleton_viz.render(
                                normalized.keypoints,
                                normalized.scores,
                                title=f"Frame {self.metrics.frame_count} | Torso: {normalized.torso_length:.1f}px"
                            )
                            cv2.imshow("ESQUELETO NORMALIZADO", normalized_canvas)
                        else:
                            # Show invalid frame indicator
                            black_canvas = np.zeros((500, 500, 3), dtype=np.uint8)
                            cv2.putText(
                                black_canvas,
                                "Frame Invalido",
                                (150, 250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                2
                            )
                            cv2.imshow("ESQUELETO NORMALIZADO", black_canvas)
                    except Exception as e:
                        print(f"[ERRO] Normalizacao: {e}")

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
            self._print_summary()
