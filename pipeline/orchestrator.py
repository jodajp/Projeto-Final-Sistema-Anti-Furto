import time
import json
import cv2
import numpy as np
import torch

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
            match_thresh=0.8, 
            #mot20=False
        )
        self.last_tracked_objects = []

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

                    bboxes_com_scores = []
                    kpts_array = np.array(keypoints) if keypoints else np.array([])
                    lista_kpts = []
                    lista_scores = []

                    # Separar as pessoas (Quer haja 1 pessoa ou 10 pessoas na loja)
                    if kpts_array.size > 0:
                        if len(kpts_array.shape) == 2: # Só 1 pessoa detetada
                            lista_kpts = [kpts_array]
                            lista_scores = [scores] if scores else [[]]
                        elif len(kpts_array.shape) == 3: # Várias pessoas detetadas
                            lista_kpts = kpts_array
                            lista_scores = scores if scores else [[] for _ in range(len(kpts_array))]

                    for scrs, kpts in zip(lista_scores, lista_kpts):
                        if len(kpts) >= 17: # Só aceita pessoas com corpo completo
                            x_min, y_min = np.min(kpts, axis=0)
                            x_max, y_max = np.max(kpts, axis=0)
                            score_medio = float(np.mean(scrs)) if len(scrs) > 0 else 1.0
                            bboxes_com_scores.append([x_min, y_min, x_max, y_max, score_medio, 0.0])

                    # Dar todas as caixas ao ByteTrack de uma só vez para ele distribuir os IDs
                    if bboxes_com_scores:
                        detections_array = torch.tensor(bboxes_com_scores, dtype=torch.float32)
                        self.last_tracked_objects = self.tracker.update(
                            detections_array, 
                            [frame.shape[0], frame.shape[1]]
                        )
                    else:
                        self.last_tracked_objects = []

                    if keypoints:
                        self.metrics.on_detection()
                elif not self.cache_result:
                    self.last_detection = ([], [])

                keypoints, scores = self.last_detection
                alert_text = None
                
                # RECONSTRUIR LISTAS PARA AVALIAÇÃO DE ATIVIDADES 
                kpts_array = np.array(keypoints) if keypoints else np.array([])
                lista_kpts = []
                lista_scores = []

                if kpts_array.size > 0:
                    if len(kpts_array.shape) == 2:
                        lista_kpts = [kpts_array]
                        lista_scores = [scores] if scores else [[]]
                    elif len(kpts_array.shape) == 3:
                        lista_kpts = kpts_array
                        lista_scores = scores if scores else [[] for _ in range(len(kpts_array))]

                #AVALIAR AS ATIVIDADES PARA CADA PESSOA SEPARADAMENTE
                for i, (scrs, kpts) in enumerate(zip(lista_scores, lista_kpts)):
                    if len(kpts) >= 17:
                        
                        # Convertemos as matrizes de volta para listas normais antes 
                        # de as entregar ao motor de atividades para ele não estoirar!
                        kpts_lista = kpts.tolist()
                        scrs_lista = scrs.tolist() if hasattr(scrs, 'tolist') else list(scrs)
                        
                        for activity in self.activities:
                            event = activity.detecta(
                                kpts_lista,
                                scrs_lista,
                                self.metrics.frame_count,
                                timestamp,
                            )
                            if event:
                                self.alert_dispatcher.dispatch(event)
                                alert_text = f"ALERTA: {event.tipo} ({event.confianca:.0%})"

                                # --- PAYLOAD JSON PARA A BASE DE DADOS ---
                                pessoa_id = "Desconhecido"
                                if i < len(self.last_tracked_objects):
                                    alvo = self.last_tracked_objects[i]
                                    # O nosso tradutor de Objetos vs Matrizes
                                    if hasattr(alvo, 'track_id'):
                                        pessoa_id = alvo.track_id
                                    else:
                                        pessoa_id = int(alvo[4])

                                relatorio_db = {
                                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp)),
                                    "track_id": pessoa_id,
                                    "tipo_alerta": event.tipo,
                                    "confianca": round(event.confianca * 100, 2)
                                }

                                payload = json.dumps(relatorio_db)
                                print(f"\n[PAYLOAD PARA A CLOUD] -> {payload}\n")

                output = self.renderer.render(frame, keypoints, scores)

                # --- 4. DESENHAR A CAIXA E O ID ---
                if len(self.last_tracked_objects) > 0:
                    for obj in self.last_tracked_objects:
                        
                        # A TUA BLINDAGEM VISUAL
                        if hasattr(obj, 'track_id'):
                            track_id = obj.track_id
                            x1, y1, x2, y2 = map(int, obj.tlbr)
                        else:
                            # Extrair da Matriz (x1, y1, x2, y2, id)
                            x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                            track_id = int(obj[4])
                        
                        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(output, f"ID: {track_id}", (x1, max(0, y1 - 10)), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

                info_lines = self._build_info_lines(had_new_inference=should_infer)
                self.renderer.draw_overlay(
                    output,
                    info_lines,
                    alert_text=alert_text,
                    debug=self.debug,
                )

                cv2.imshow(self.renderer.window_name, output)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("d"), ord("D")):
                    self.debug = not self.debug
                    state = "ATIVADO" if self.debug else "DESATIVADO"
                    print(f"[DEBUG] {state}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._print_summary()