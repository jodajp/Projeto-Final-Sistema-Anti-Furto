import os
from abc import ABC, abstractmethod

import cv2

class BaseVideoSource(ABC):
    """Classe base abstrata para todas as fontes de vídeo."""
    @abstractmethod
    def open(self) -> cv2.VideoCapture:
        pass

class CameraSource(BaseVideoSource):
    """Lida exclusivamente com a camera ativada"""
    def __init__(self, config: dict):
        self.config = config

    def open(self) -> cv2.VideoCapture:
        # Garante que o ID da câmara é um inteiro
        source = int(self.config.get("id", 0))
        backend_name = self.config.get("backend", "CAP_DSHOW")
        backend = getattr(cv2, backend_name, cv2.CAP_DSHOW)

        if isinstance(source, str):
            if source.isdigit():
                source = int(source)
            else:
                # It is a video file path, use default ANY backend
                backend = cv2.CAP_ANY

        cap = cv2.VideoCapture(source, backend)

        # Força a resolução e FPS
        width = int(self.config.get("width", 640))
        height = int(self.config.get("height", 480))
        fps = int(self.config.get("fps", 30))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            raise RuntimeError(f"Erro: Não foi possível abrir a câmara com ID: {source}")
        
        print(f"[CAMERA] Fonte {source} iniciada ({width}x{height} @ {fps}fps)")
        return cap

class FileSource(BaseVideoSource):
    """Lida exclusivamente com ficheiros de vídeo (.mp4, .avi, etc)."""
    def __init__(self, config: dict):
        self.config = config

    def open(self) -> cv2.VideoCapture:
        source = str(self.config.get("id", ""))
        # Não passamos backend nem forçamos resolução para proteger o codec
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Erro: Não foi possível carregar o vídeo em: {source}")
        
        # Lê os metadados reais do ficheiro
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[VIDEO] Lendo ficheiro: {os.path.basename(source)}")
        print(f"        Resolução original: {w}x{h} | FPS: {fps:.2f}")
        
        return cap

def create_video_source(config: dict) -> BaseVideoSource:
    """Fábrica que devolve a classe correta consoante o input."""
    source = config.get("id", 0)
    
    # Se for string e não for apenas números, assumimos que é um caminho de ficheiro
    if isinstance(source, str) and not source.isdigit():
        return FileSource(config)
    
    # Caso contrário, é uma câmara
    return CameraSource(config)
