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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

config = AppConfig.from_file("config.yaml")

# Dicionário para gerir sessões individuais
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

def get_orchestrator(sid):
    if sid not in sessions:
        # Criamos um orchestrator novo para cada utilizador que se liga
        sessions[sid] = AntiTheftOrchestrator(config)
    return sessions[sid]

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in sessions:
        # Tentar libertar os recursos da memória/GPU
        orchestrator = sessions[request.sid]
        if hasattr(orchestrator, 'cleanup'):
            try:
                orchestrator.cleanup()
            except Exception as e:
                print(f"[ERRO] Falha ao limpar recursos do orchestrator: {e}")
                
        del sessions[request.sid]
        print(f"[INFO] Sessão limpa para: {request.sid}")

@socketio.on('frame')
def handle_frame(binary_frame):
    sid = request.sid
    orchestrator = get_orchestrator(sid)
    metrics = orchestrator.metrics

    # Converter bytes diretamente para imagem OpenCV e tratar possíveis erros
    try:
        np_arr = np.frombuffer(binary_frame, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERRO] Falha ao descodificar a imagem: {e}")
        return

    if frame is None:
        return

    metrics.on_frame()
    timestamp = time.time()
    
    # Lógica de Inferência (mantida conforme o original)
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

    # Renderização
    output = orchestrator.renderer.render(frame, keypoints, scores)
    
    # Codificar para JPEG e enviar como BINÁRIO
    ret, buffer = cv2.imencode('.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ret:
        # Enviar bytes brutos de volta ao browser
        emit('response_frame', buffer.tobytes())

if __name__ == '__main__':
    # Usar eventlet ou gevent se possível para melhor performance assíncrona
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)