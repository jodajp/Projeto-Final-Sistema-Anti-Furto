import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, request, render_template
from flask_socketio import SocketIO, emit

# Adiciona a raiz do projeto ao PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AppConfig
from pipeline.orchestrator import AntiTheftOrchestrator

class AntiTheftServer:
    def __init__(self, config_path="config.yaml"):
        """Inicializa a aplicação Flask, SocketIO e carrega as configurações."""
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.config = AppConfig.from_file(config_path)
        
        # Dicionário para gerir sessões (1 câmara aberta = 1 orchestrator)
        self.sessions = {}

        # Regista rotas HTTP e bindings de Socket.IO
        self._register_handlers()

    def _register_handlers(self):
        """Mapeia as rotas e eventos aos métodos descritos abaixo."""
        self.app.route('/')(self.index)
        self.socketio.on('disconnect')(self.handle_disconnect)
        self.socketio.on('frame')(self.handle_frame)

    def index(self):
        """Rota de fallback caso acedam diretamente pelo browser nativo ao Flask."""
        return render_template('index.html')

    def get_orchestrator(self, sid):
        """Retorna ou cria uma instância do orchestrator (Modelo de IA) em tempo real para a ligação."""
        if sid not in self.sessions:
            self.sessions[sid] = AntiTheftOrchestrator(self.config)
        return self.sessions[sid]

    def handle_disconnect(self):
        """Processa as limpezas de memória gráfica quando o utilizador pára a câmara com o Vue.js."""
        sid = request.sid
        if sid in self.sessions:
            orchestrator = self.sessions[sid]
            if hasattr(orchestrator, 'cleanup'):
                try:
                    orchestrator.cleanup()
                except Exception as e:
                    print(f"[ERRO] Falha ao limpar recursos do orchestrator: {e}")
                    
            del self.sessions[sid]
            print(f"[INFO] Sessão limpa e recursos libertados para: {sid}")

    def handle_frame(self, binary_frame):
        """Método pesado que injeta Frames da Câmara Web no Modelo MMPose/Orchestrator."""
        sid = request.sid
        orchestrator = self.get_orchestrator(sid)
        metrics = orchestrator.metrics

        # Converter bytes transmitidos do Vue em imagem para manipulação de CV2
        try:
            np_arr = np.frombuffer(binary_frame, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[ERRO] Falha ao descodificar a imagem de {sid}: {e}")
            return

        if frame is None:
            return

        metrics.on_frame()
        timestamp = time.time()
        
        # Lógica de Inferência
        should_infer = (metrics.frame_count % metrics.frame_skip) == 0
        if should_infer:
            detection = orchestrator.detector.detect(frame)
            keypoints, scores = ([], [])
            if isinstance(detection, tuple) and len(detection) == 2:
                keypoints = list(detection[0]) if detection[0] is not None else []
                scores = list(detection[1]) if detection[1] is not None else []
            orchestrator.last_detection = (keypoints, scores)
        
        keypoints, scores = orchestrator.last_detection

        # Processamento de Atividades e Alertas
        if keypoints:
            for activity in orchestrator.activities:
                event = activity.detecta(keypoints, scores, metrics.frame_count, timestamp)
                if event:
                    orchestrator.alert_dispatcher.dispatch(event)

        # Renderização visual (Desenhar pontos na imagem)
        output = orchestrator.renderer.render(frame, keypoints, scores)
        
        # Codificar para JPEG e enviar devolutamente como BINÁRIO ao Front-End Vue
        ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            emit('response_frame', buffer.tobytes())

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Inicia efetivamente as comunicações da Web App."""
        print(f"[INFO] A iniciar backend AntiTheftServer (Porta de Escuta: {port})")
        self.socketio.run(self.app, host=host, port=port, debug=debug)

if __name__ == '__main__':
    server = AntiTheftServer()
    server.run()
