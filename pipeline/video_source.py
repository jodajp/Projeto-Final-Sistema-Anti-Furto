import cv2

# Class para lidar com a fonte de vídeo agora penas camera, expansão para fotos ou vídeos pode ser feita futuramente
class VideoSource:
    def __init__(self, config: dict):
        self.config = config

    def open(self):
        source = self.config.get("id", 0)
        
        backend_name = self.config.get("backend", "CAP_DSHOW")
        backend = getattr(cv2, backend_name, cv2.CAP_DSHOW)

        if isinstance(source, str):
            if source.isdigit():
                source = int(source)
            else:
                # It is a video file path, use default ANY backend
                backend = cv2.CAP_ANY

        cap = cv2.VideoCapture(source, backend)

        width = int(self.config.get("width", 640))
        height = int(self.config.get("height", 480))
        fps = int(self.config.get("fps", 30))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir camera/fonte: {source}")

        return cap
