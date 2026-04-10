import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, request
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
    return """
    <html>
        <head>
            <title>Sistema Anti-Furto</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.1/socket.io.js"></script>
        </head>
        <body style="text-align: center; background-color: #333; color: white;">
            <h1>Sistema Anti-Furto - Monitorização pelo Browser</h1>
            
            <video id="webcam_video" autoplay playsinline style="display:none;"></video>

            <canvas id="hidden_canvas" style="display:none;"></canvas>
            
            <!-- Imagem Final com a renderização dos alertas -->
            <img id="processed_img" width="800" style="border: 2px solid white;" />
            <br>
            <button id="start-btn" style="padding: 10px; margin-top:20px;">Iniciar Transmissão</button>
            <button id="stop-btn" style="padding: 10px; margin-top:20px; display: none;">Parar</button>
            
            <script>
                const socket = io();
                const video = document.getElementById('webcam_video');
                const canvas = document.getElementById('hidden_canvas');
                const ctx = canvas.getContext('2d');
                const img = document.getElementById('processed_img');
                const startBtn = document.getElementById('start-btn');
                
                let streamIniciado = false;
                let localStream = null;
                let aguardandoResposta = false; // Controlo de fluxo

                startBtn.onclick = async () => {
                    try {
                        localStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                        video.srcObject = localStream;
                        video.play();
                        streamIniciado = true;
                        startBtn.style.display = 'none';

                        video.onloadedmetadata = () => {
                            canvas.width = 640; 
                            canvas.height = 480;
                            enviarFrame(); // Inicia o ciclo
                        };
                    } catch (err) {
                        alert("Erro na câmara: " + err);
                    }
                };

                function enviarFrame() {
                    if (!streamIniciado) return;

                    // Se o servidor ainda está a processar o frame anterior, não enviamos um novo (evita lag)
                    if (!aguardandoResposta) {
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        
                        // Converte para Binário (Blob) em vez de String Base64
                        canvas.toBlob((blob) => {
                            socket.emit('frame', blob);
                            aguardandoResposta = true; 
                        }, 'image/jpeg', 0.7);
                    }
                    
                    // Mantém os 10-15 FPS, mas respeita o tempo de resposta do servidor
                    setTimeout(enviarFrame, 100); 
                }

                // Recebe o buffer binário e converte para URL de imagem
                socket.on('response_frame', (data) => {
                    const blob = new Blob([data], { type: 'image/jpeg' });
                    const url = URL.createObjectURL(blob);
                    
                    // Liberta a memória da URL anterior para não crashar o browser
                    const oldUrl = img.src;
                    img.src = url;
                    if (oldUrl.startsWith('blob:')) URL.revokeObjectURL(oldUrl);
                    
                    aguardandoResposta = false;
                });
            </script>
        </body>
    </html>
    """

def get_orchestrator(sid):
    if sid not in sessions:
        # Criamos um orchestrator novo para cada utilizador que se liga
        sessions[sid] = AntiTheftOrchestrator(config)
    return sessions[sid]

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in sessions:
        del sessions[request.sid]
        print(f"[INFO] Sessão limpa para: {request.sid}")

@socketio.on('frame')
def handle_frame(binary_frame):
    sid = request.sid
    orchestrator = get_orchestrator(sid)
    metrics = orchestrator.metrics

    # Converter bytes diretamente para imagem OpenCV (Muito mais rápido que Base64)
    np_arr = np.frombuffer(binary_frame, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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