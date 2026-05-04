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
            match_thresh=0.8, 
            #mot20=False
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
                            
                            # 1. Encontrar os limites do esqueleto
                            x_min, y_min = np.min(kpts, axis=0)
                            x_max, y_max = np.max(kpts, axis=0)
                            
                            # 2. Adicionar margem (padding) para a caixa cobrir o corpo inteiro
                            margem_x = 25
                            margem_y = 35
                            
                            # 3. Limitar a matemática às bordas do ecrã (0 a 640x480)
                            x_min = max(0, x_min - margem_x)
                            y_min = max(0, y_min - margem_y)
                            x_max = min(frame.shape[1], x_max + margem_x) 
                            y_max = min(frame.shape[0], y_max + margem_y) 
                            
                            score_medio = float(np.mean(scrs)) if len(scrs) > 0 else 1.0
                            bboxes_com_scores.append([x_min, y_min, x_max, y_max, score_medio, 0.0])

                    # Dar todas as caixas ao ByteTrack de uma só vez para ele distribuir os IDs
                    if bboxes_com_scores:
                        detections_array = torch.tensor(bboxes_com_scores, dtype=torch.float32)
                        
                        res_altura = frame.shape[0] # 480
                        res_largura = frame.shape[1] # 640
                        
                        self.last_tracked_objects = self.tracker.update(
                            detections_array, 
                            [res_altura, res_largura] 
                        )
                    else:
                        self.last_tracked_objects = []

                    if keypoints:
                        self.metrics.on_detection()
                elif not self.cache_result:
                    self.last_detection = ([], [])

                keypoints, scores = self.last_detection
                alert_text = None
                
                # --- 2. O SISTEMA DE BLINDAGEM VISUAL (Bypass ao Tracker) ---
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

                # A. Construir as tuas caixas perfeitas intocáveis
                pessoas_originais = []
                for scrs, kpts in zip(lista_scores, lista_kpts):
                    if len(kpts) >= 17:
                        x_min, y_min = np.min(kpts, axis=0)
                        x_max, y_max = np.max(kpts, axis=0)
                        
                        # A tua margem perfeita
                        x1 = int(max(0, x_min - 25))
                        y1 = int(max(0, y_min - 35))
                        x2 = int(min(frame.shape[1], x_max + 25))
                        y2 = int(min(frame.shape[0], y_max + 35))
                        
                        pessoas_originais.append({
                            'kpts': kpts.tolist(),
                            'scrs': scrs.tolist() if hasattr(scrs, 'tolist') else list(scrs),
                            'box': (x1, y1, x2, y2),
                            'center_x': (x1 + x2) / 2
                        })

                # B. Extrair apenas os IDs do tracker ignorando o lixo matemático dele
                objetos_tracker = []
                for obj in self.last_tracked_objects:
                    if hasattr(obj, 'track_id'):
                        track_id = obj.track_id
                        tx1, tx2 = float(obj.tlbr[0]), float(obj.tlbr[2])
                    else:
                        tx1, tx2 = float(obj[0]), float(obj[2])
                        track_id = int(obj[4])
                    objetos_tracker.append({'id': track_id, 'center_x': (tx1 + tx2) / 2})

                # C. Ordenar ambos da esquerda para a direita para casar o ID correto
                pessoas_originais.sort(key=lambda p: p['center_x'])
                objetos_tracker.sort(key=lambda t: t['center_x'])

                for i, pessoa in enumerate(pessoas_originais):
                    pessoa['id'] = objetos_tracker[i]['id'] if i < len(objetos_tracker) else "Desconhecido"

                # --- 3. AVALIAR ATIVIDADES E CRIAR JSON ---
                for pessoa in pessoas_originais:
                    for activity in self.activities:
                        event = activity.detecta(
                            pessoa['kpts'],
                            pessoa['scrs'],
                            self.metrics.frame_count,
                            timestamp,
                        )
                        if event:
                            self.alert_dispatcher.dispatch(event)
                            alert_text = f"ALERTA: {event.tipo} ({event.confianca:.0%})"

                            # --- AQUI ENTRA A BASE DE DADOS ---
                            # Guardamos os dados de forma limpa
                            self.db.salvar_alerta(
                                track_id=pessoa['id'],
                                tipo_alerta=event.tipo,
                                confianca=event.confianca * 100 # Guardamos como percentagem 0-100
                            )

                            # Manter o teu print para debug se quiseres
                            print(f"[DB] Registado: ID {pessoa['id']} a fazer {event.tipo}")

                output = self.renderer.render(frame, keypoints, scores)

                # --- 4. DESENHAR AS CAIXAS PERFEITAS NO ECRÃ ---
                for pessoa in pessoas_originais:
                    x1, y1, x2, y2 = pessoa['box']
                    track_id = pessoa['id']
                    
                    cv2.rectangle(output, (x1 // 2, y1 // 2), (x2 // 2, y2 // 2), (0, 255, 0), 3)
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