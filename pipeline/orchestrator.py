import os
import time
import json
import cv2
import numpy as np
import torch

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

        if kpts_arr.size == 0 or kpts_arr.shape[-1] != 2:
            return []

        viz_cfg = self.config.visualization()
        padding = viz_cfg.get('bbox_padding', {'x': 25, 'y': 35})
        conf_min = viz_cfg.get('confidence_threshold', 0.3)
        class_id = viz_cfg.get('default_class_id', 0.0)

        entidades = []
        for scrs, kpts in zip(scrs_arr, kpts_arr):
            mask = scrs > conf_min
            if not np.any(mask): continue 

            kpts_vis = kpts[mask]
            x_min, y_min = np.min(kpts_vis, axis=0)/2
            x_max, y_max = np.max(kpts_vis, axis=0)/2

            x1 = int(max(0, x_min - padding['x']))
            y1 = int(max(0, y_min - padding['y']))
            x2 = int(min(frame_shape[1], x_max + padding['x']))
            y2 = int(min(frame_shape[0], y_max + padding['y']))

            entidades.append({
                'kpts': kpts.tolist(),
                'scrs': scrs.tolist(),
                'box': [x1, y1, x2, y2, float(np.mean(scrs[mask])), class_id],
                'center_x': (x1 + x2) / 2,
                'id': "Desconhecido"
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
                if self.metrics.frame_count == 0:
                    print(f"[VIDEO] Resolução Real: {frame.shape[1]}x{frame.shape[0]}")

                self.metrics.on_frame()
                timestamp = time.time()

                # 1. INFERÊNCIA OU CACHE
                should_infer = (self.metrics.frame_count % self.metrics.frame_skip) == 0
                if should_infer:
                    t0 = time.time()
                    keypoints, scores = self.detector.detect(frame)
                    self.last_inference_ms = (time.time() - t0) * 1000.0
                    self.metrics.on_inference(self.last_inference_ms)
                    self.last_detection = (keypoints, scores)
                else:
                    keypoints, scores = self.last_detection

                # 2. PROCESSAMENTO ÚNICO (Reciclagem de código)
                entidades = self._preparar_entidades(keypoints, scores, frame.shape)

                # 3. ATUALIZAÇÃO DO TRACKER
                if should_infer:
                    if entidades:
                        # Extrai apenas as bboxes validadas para o tracker
                        bboxes = torch.tensor([e['box'] for e in entidades], dtype=torch.float32)
                        self.last_tracked_objects = self.tracker.update(bboxes, frame.shape[:2])
                        self.metrics.on_detection()
                    else:
                        self.last_tracked_objects = []

                # 4. CASAMENTO DE IDS E LÓGICA
                self._atribuir_ids(entidades)
                alert_text = self._processar_atividades(entidades, timestamp)

                # 5. RENDERIZAÇÃO FINAL
                output = self.renderer.render(frame, keypoints, scores)
                self._desenhar_caixas(output, entidades)

                self.renderer.draw_overlay(output, self._build_info_lines(should_infer), alert_text, self.debug)
                cv2.imshow(self.renderer.window_name, output)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")): break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_summary()