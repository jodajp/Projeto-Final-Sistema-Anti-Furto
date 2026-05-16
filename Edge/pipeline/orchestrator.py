import time
import cv2
import numpy as np
import torch

from Alertas.database_handler import DatabaseHandler
from bytetracker import BYTETracker
from Edge.Detecao.detector_factory import create_detector
from .activity_loader import load_activities
from .alert_dispatcher import AlertDispatcher, load_alert_handlers
from .metrics import PipelineMetrics
from .renderer import PoseRenderer
from .video_source import create_video_source
from .spatial_normalizer import SpatialNormalizer
from .skeleton_visualizer import SkeletonVisualizer
from .temporal_pose_filter import TemporalPoseFilter


class AntiTheftOrchestrator:
    """Coordena detector, atividades, alertas, render e loop de video."""

    def __init__(self, config):
        self.config = config
        self.runtime_config = config.runtime()

        # Inicializa o detector de pose
        self.detector = create_detector(config.detector_config())
        self.detector_info = self.detector.get_info() or {}

        self.activities = load_activities(config.activity_specs())
        handlers = load_alert_handlers(config.alert_specs())
        self.alert_dispatcher = AlertDispatcher(handlers)

        self.renderer = PoseRenderer(config.visualization())
        self.video_source = create_video_source(config.camera())
        self.metrics = PipelineMetrics(frame_skip=config.frame_skip())

        self.cache_result = bool(self.runtime_config.get("cache_result", True))
        self.debug = bool(self.runtime_config.get("debug", False))

        # Initialize temporal pose filtering directly from config
        self.temporal_filter = TemporalPoseFilter(config)
        self.temporal_filters_por_id = {}

        # Initialize spatial normalization
        spatial_cfg = config.data.get("spatial_normalization", {})
        if spatial_cfg.get("enabled", False):
            self.spatial_norm_enabled = True
            self.normalizer = SpatialNormalizer(config)
            self.skeleton_viz = SkeletonVisualizer()
            self.show_normalized = True
        else:
            self.spatial_norm_enabled = False
            self.normalizer = None
            self.skeleton_viz = None
            self.show_normalized = False

        self.last_detection = ([], [])
        self.last_inference_ms = 0.0
        self.frame = None

        # Tuned tracker params for more persistent IDs
        self.tracker = BYTETracker(
            track_thresh=0.4,
            track_buffer=60,
            match_thresh=0.6,
        )
        self.last_tracked_objects = []

        self.db = DatabaseHandler()
        # debug helper to print bbox/keypoint diagnostics once
        self._debug_bbox_printed = False


    def _print_startup(self):
        print("\n" + "=" * 60)
        print("SISTEMA ANTI-FURTO - MODULAR PIPELINE")
        print("=" * 60)
        print(f"Backend: {self.detector_info.get('backend', 'N/A')}")
        print(f"Atividades carregadas: {len(self.activities)}")
        print(f"Handlers de alerta: {len(self.alert_dispatcher.handlers)}")
        temporal_status = "✓ ATIVO" if self.temporal_filter.is_enabled() else "desativado"
        print(f"Filtro Temporal: {temporal_status}")
        spatial_status = "✓ ATIVO" if self.spatial_norm_enabled else "desativado"
        print(f"Normalização Espacial: {spatial_status}")
        print("Controles: Q = sair | D = debug | N = toggle normalizado | T = toggle temporal")
        print("=" * 60 + "\n")

    def _count_people(self, keypoints):
        if keypoints is None:
            return 0

        keypoints_array = np.asarray(keypoints)
        if keypoints_array.size == 0:
            return 0
        if keypoints_array.ndim == 2:
            return 1
        if keypoints_array.ndim == 3:
            return int(keypoints_array.shape[0])
        return 0

    def _build_info_lines(self, had_new_inference: bool):
        people_count = self._count_people(self.last_detection[0])
        inference_state = "NEW" if had_new_inference else "CACHED"

        return [
            f"FPS: {self.metrics.fps:.1f}",
            f"Frame: {self.metrics.frame_count}",
            f"People: {people_count}",
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

        h_img, w_img = frame_shape[:2]
        viz_cfg = self.config.visualization()
        padding = viz_cfg.get('bbox_padding', {'x': 25, 'y': 35})
        conf_min = viz_cfg.get('confidence_threshold', 0.3)

        entidades = []
        for scrs, kpts in zip(scrs_arr, kpts_arr):
            # Filter keypoints: must have valid coordinates and confidence > 0.2
            valid_mask = (scrs > 0.2) & np.isfinite(kpts).all(axis=1)
            valid_mask = valid_mask & (kpts[:, 0] > 0) & (kpts[:, 0] < w_img)
            valid_mask = valid_mask & (kpts[:, 1] > 0) & (kpts[:, 1] < h_img)
            
            if not np.any(valid_mask):
                continue
            
            kpts_valid = kpts[valid_mask]
            x_min, y_min = np.min(kpts_valid, axis=0)
            x_max, y_max = np.max(kpts_valid, axis=0)

            x1 = int(np.clip(x_min - padding['x'], 0, w_img - 1))
            y1 = int(np.clip(y_min - padding['y'], 0, h_img - 1))
            x2 = int(np.clip(x_max + padding['x'], 0, w_img - 1))
            y2 = int(np.clip(y_max + padding['y'], 0, h_img - 1))

            # Diagnostic print to help debug bbox placement (once)
            if not getattr(self, '_debug_bbox_printed', False):
                try:
                    print(f"[DEBUG_BBOX] frame_shape=(w={w_img},h={h_img}), valid_kpts_min=({x_min:.1f},{y_min:.1f}), max=({x_max:.1f},{y_max:.1f}), box=({x1},{y1},{x2},{y2})")
                except Exception:
                    pass
                self._debug_bbox_printed = True

            entidades.append({
                'kpts': kpts.tolist(),
                'scrs': scrs.tolist(),
                'box': [x1, y1, x2, y2, float(np.mean(scrs[valid_mask])), 0.0],
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                'id': '...'
            })
        return entidades

    def _atribuir_ids(self, entidades):
        """Atribui IDs do ByteTrack às entidades usando distância 2D para melhor rastreamento."""
        if not entidades or self.last_tracked_objects is None or len(self.last_tracked_objects) == 0:
            return

        objetos_tracker = []
        for obj in self.last_tracked_objects:
            if hasattr(obj, 'track_id'):
                track_id = obj.track_id
                tx1, tx2, ty1, ty2 = float(obj.tlbr[0]), float(obj.tlbr[2]), float(obj.tlbr[1]), float(obj.tlbr[3])
            else:
                tx1, tx2, ty1, ty2, track_id = float(obj[0]), float(obj[2]), float(obj[1]), float(obj[3]), int(obj[4])
            cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            objetos_tracker.append({'id': track_id, 'center_x': cx, 'center_y': cy})

        # Dynamic max distance threshold based on frame width
        if self.frame is not None and hasattr(self.frame, 'shape'):
            try:
                frame_w = int(self.frame.shape[1])
            except Exception:
                frame_w = 640
        else:
            frame_w = 640
        max_match_dist = max(80, int(frame_w * 0.25))

        # Greedy matching: assign nearest tracker to each entity and remove matched tracker
        remaining = objetos_tracker.copy()
        for entidade in entidades:
            ex, ey = entidade['center_x'], entidade['center_y']
            if not remaining:
                continue
            # find nearest
            closest = min(remaining, key=lambda t: (ex - t['center_x'])**2 + (ey - t['center_y'])**2)
            dist_sq = (ex - closest['center_x'])**2 + (ey - closest['center_y'])**2
            if dist_sq < max_match_dist**2:
                entidade['id'] = closest['id']
                # remove to prevent duplicate assignment
                remaining = [r for r in remaining if r['id'] != closest['id']]

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
            
    def _processar_frame(self, keypoints, scores, frame_shape, timestamp):
        """Processa poses, tracking, filtro temporal e atividades."""
        if keypoints is None or scores is None:
            return [], None

        entidades = self._preparar_entidades(keypoints, scores, frame_shape)
        if not entidades:
            self.last_tracked_objects = []
            return [], None

        bboxes = torch.tensor([ent['box'] for ent in entidades], dtype=torch.float32)
        self.last_tracked_objects = self.tracker.update(bboxes, frame_shape[:2])
        self.metrics.on_detection()
        self._atribuir_ids(entidades)

        if self.temporal_filter.is_enabled():
            ids_presentes = {ent['id'] for ent in entidades if ent.get('id') is not None}
            self.temporal_filters_por_id = {
                track_id: filter_obj
                for track_id, filter_obj in self.temporal_filters_por_id.items()
                if track_id in ids_presentes
            }

            for ent in entidades:
                track_id = ent.get('id')
                if track_id is None:
                    continue

                filter_obj = self.temporal_filters_por_id.get(track_id)
                if filter_obj is None:
                    filter_obj = TemporalPoseFilter(self.config)
                    self.temporal_filters_por_id[track_id] = filter_obj

                kpts_arr = np.asarray(ent['kpts'], dtype=np.float32)
                scrs_arr = np.asarray(ent['scrs'], dtype=np.float32)
                filt_kpts, filt_scrs, _ = filter_obj.filter_pose(kpts_arr, scrs_arr)
                ent['kpts'] = filt_kpts.tolist()
                ent['scrs'] = filt_scrs.tolist()

        alert_text = self._processar_atividades(entidades, timestamp)
        return entidades, alert_text


    def _render_normalized_canvas(self, entidades):
        if not entidades or not self.spatial_norm_enabled:
            return None
        if self.normalizer is None or self.skeleton_viz is None:
            return None

        poses = []
        for ent in entidades[:4]:
            kpts_arr = np.asarray(ent['kpts'], dtype=np.float32)
            scrs_arr = np.asarray(ent['scrs'], dtype=np.float32)
            if kpts_arr.shape != (17, 2) or scrs_arr.shape != (17,):
                continue

            try:
                normalized = self.normalizer.normalize(kpts_arr, scrs_arr)
            except Exception as exc:
                if self.debug:
                    print(f"[ERRO] Normalizacao: {exc}")
                continue

            if normalized.is_valid:
                title = f"ID {ent.get('id', '?')} | Torso: {normalized.torso_length:.1f}px"
                canvas = self.skeleton_viz.render(normalized.keypoints, normalized.scores, title=title)
            else:
                canvas = np.zeros((self.skeleton_viz.canvas_size, self.skeleton_viz.canvas_size, 3), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    f"ID {ent.get('id', '?')} - Frame invalido",
                    (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            poses.append(canvas)

        if not poses:
            return None

        cell_size = self.skeleton_viz.canvas_size // 2
        grid = np.zeros((self.skeleton_viz.canvas_size, self.skeleton_viz.canvas_size, 3), dtype=np.uint8)
        for idx, canvas in enumerate(poses[:4]):
            row = idx // 2
            col = idx % 2
            y1 = row * cell_size
            x1 = col * cell_size
            resized = cv2.resize(canvas, (cell_size, cell_size))
            grid[y1:y1 + cell_size, x1:x1 + cell_size] = resized

        return grid

    def run(self):
        self._print_startup()
        cap = self.video_source.open()
        try:
            while True:
                ret, frame = cap.read()
                self.frame = frame
                if not ret:
                    break

                if self.metrics.frame_count == 0:
                    print(f"[VIDEO] Resolução Real: {frame.shape[1]}x{frame.shape[0]}")

                self.metrics.on_frame()
                timestamp = time.time()

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

                entidades = []
                alert_text = None
                if keypoints is not None and len(keypoints) > 0:
                    entidades, alert_text = self._processar_frame(keypoints, scores, frame.shape, timestamp)

                # Render using processed entities so skeletons align with boxes/IDs
                if entidades:
                    render_kpts = [ent['kpts'] for ent in entidades]
                    render_scrs = [ent['scrs'] for ent in entidades]
                else:
                    render_kpts, render_scrs = keypoints, scores

                output = self.renderer.render(frame, render_kpts, render_scrs)
                info_lines = self._build_info_lines(had_new_inference=should_infer)
                self._desenhar_caixas(output, entidades)
                self.renderer.draw_overlay(output, info_lines, alert_text=alert_text, debug=self.debug)

                # Show three simple windows: main video, skeleton-only, normalized
                cv2.imshow(self.renderer.window_name, output)

                sk_canvas = getattr(self.renderer, 'last_canvas', None)
                if sk_canvas is not None:
                    try:
                        cv2.imshow(self.renderer.window_name + ' - SKELETON', sk_canvas)
                    except Exception:
                        pass

                if self.spatial_norm_enabled and self.show_normalized:
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
                    state = "ATIVADO" if self.debug else "DESATIVADO"
                    print(f"[DEBUG] {state}")
                if key in (ord("n"), ord("N")) and self.spatial_norm_enabled:
                    self.show_normalized = not self.show_normalized
                    state = "VISIVEL" if self.show_normalized else "OCULTO"
                    print(f"[NORMALIZADO] {state}")
                if key in (ord("t"), ord("T")):
                    self.temporal_filter.toggle()
                    if not self.temporal_filter.is_enabled():
                        self.temporal_filters_por_id.clear()
                    state = "ATIVO" if self.temporal_filter.is_enabled() else "INATIVO"
                    print(f"[TEMPORAL] Filtro {state}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_summary()
