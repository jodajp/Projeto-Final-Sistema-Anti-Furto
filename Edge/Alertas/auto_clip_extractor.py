"""
Auto Clip Extractor Plugin
Monitoriza continuamente os frames para cada track_id e, quando ocorre um evento,
cria uma sessão de gravação contínua que inclui frames pré-evento e pós-evento.
O clipe final contém apenas as conexões do esqueleto num fundo preto.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Dict, List, Any

from Edge.Atividades_Suspeitas.base_activity import SuspiciousEvent
from Detecao.skeleton import SKELETON_CONNECTIONS

class RecordingSession:
    """Representa uma gravação contínua para um track_id."""
    def __init__(self, pessoa_id: int, tipo_evento: str, start_frame: int):
        self.pessoa_id = pessoa_id
        self.tipo_evento = tipo_evento
        self.frames_kpts: List[np.ndarray] = []
        self.end_frame_target = start_frame  # Atualizado via extends
        self.closed = False


class AutoClipExtractor:
    """Plugin para extrair clipes de esqueletos de forma contínua."""
    
    def __init__(
        self,
        pasta_clips: str = './Alertas/clips',
        largura: int = 640,
        altura: int = 480,
        fps: float = 30.0,
        pre_event_frames: int = 60,
        post_event_frames: int = 60
    ):
        self.pasta_clips = Path(pasta_clips)
        self.pasta_clips.mkdir(parents=True, exist_ok=True)
        self.largura = largura
        self.altura = altura
        self.fps = fps
        self.pre_event_frames = pre_event_frames
        self.post_event_frames = post_event_frames
        
        # Estado
        self.buffers: Dict[int, deque] = {}         # track_id -> deque de kpts
        self.sessions: Dict[int, RecordingSession] = {} # track_id -> RecordingSession
        
    def processa_frame(self, entidades: List[Dict[str, Any]], frame_shape: tuple, frame_id: int):
        """Hook chamado pelo orchestrator para cada frame processado."""
        
        # Atualiza a resolução dinamicamente a partir do frame original
        if frame_shape and len(frame_shape) >= 2:
            self.altura = int(frame_shape[0])
            self.largura = int(frame_shape[1])
            
        ids_presentes = set()
        
        for ent in entidades:
            track_id = ent.get('id')
            if track_id is None or track_id == '...':
                continue
                
            ids_presentes.add(track_id)
            kpts = np.asarray(ent['kpts'], dtype=np.float32)
            
            # Atualizar buffer pré-evento
            if track_id not in self.buffers:
                self.buffers[track_id] = deque(maxlen=self.pre_event_frames)
            self.buffers[track_id].append(kpts)
            
            # Se houver uma sessão ativa, adiciona este frame à sessão
            if track_id in self.sessions:
                session = self.sessions[track_id]
                if not session.closed:
                    session.frames_kpts.append(kpts)
                    
                    # Verifica se atingimos o timeout da sessão
                    if frame_id >= session.end_frame_target:
                        self._fechar_sessao(session)
                        del self.sessions[track_id]
                        
        # Limpar buffers/sessões de track_ids perdidos
        for track_id in list(self.buffers.keys()):
            if track_id not in ids_presentes:
                # O track_id desapareceu
                del self.buffers[track_id]
                
                # Fechar a sessão ativa, se existir
                if track_id in self.sessions:
                    self._fechar_sessao(self.sessions[track_id])
                    del self.sessions[track_id]
        
    def registra_evento(self, evento: SuspiciousEvent, verbose: bool = True):
        """Inicia ou extende uma sessão de gravação para o pessoa_id."""
        track_id = evento.pessoa_id
        if track_id is None:
            return
            
        current_frame = evento.frame_id
        target_end_frame = current_frame + self.post_event_frames
        
        if track_id in self.sessions:
            # Atividade continua: estender o timeout
            self.sessions[track_id].end_frame_target = target_end_frame
            if verbose:
                print(f"[AutoClipExtractor] Sessão estendida para track {track_id} (timeout = {target_end_frame})")
        else:
            # Nova atividade: criar sessão e carregar o buffer pré-evento
            session = RecordingSession(track_id, evento.tipo, target_end_frame)
            session.end_frame_target = target_end_frame
            
            if track_id in self.buffers:
                # Carrega o histórico
                session.frames_kpts.extend(list(self.buffers[track_id]))
                
            self.sessions[track_id] = session
            if verbose:
                print(f"[AutoClipExtractor] Sessão iniciada para track {track_id} (timeout = {target_end_frame})")
                
    def _fechar_sessao(self, session: RecordingSession):
        """Finaliza a sessão, renderiza o vídeo e liberta memória."""
        session.closed = True
        if not session.frames_kpts:
            return
            
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{session.tipo_evento}_track{session.pessoa_id}_{timestamp_str}.mp4"
        filepath = str(self.pasta_clips / filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (self.largura, self.altura))
        
        try:
            for kpts in session.frames_kpts:
                frame = np.zeros((self.altura, self.largura, 3), dtype=np.uint8)
                
                # Desenhar as ligações do esqueleto (linhas)
                for j1, j2 in SKELETON_CONNECTIONS:
                    x1, y1 = kpts[j1][0], kpts[j1][1]
                    x2, y2 = kpts[j2][0], kpts[j2][1]
                    
                    if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                        continue
                        
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
                        
                # Desenhar as articulações (pontos)
                for pt in kpts:
                    x, y = pt[0], pt[1]
                    if np.isnan(x) or np.isnan(y):
                        continue
                    p = (int(x), int(y))
                    if p[0] > 0 and p[1] > 0:
                        cv2.circle(frame, p, 3, (0, 255, 0), -1)
                        
                out.write(frame)
                
            print(f"\n[AutoClipExtractor] Clipe contínuo de {len(session.frames_kpts)} frames salvo em: {filepath}\n")
            
        except Exception as e:
            print(f"[ERRO AutoClipExtractor] Falha ao gerar o clipe contínuo: {e}")
        finally:
            out.release()
