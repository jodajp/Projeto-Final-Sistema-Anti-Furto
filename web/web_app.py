import os
import sys
import time
import cv2
from flask import Flask, Response

# Adiciona a raiz do projeto ao PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AppConfig
from pipeline.orchestrator import AntiTheftOrchestrator

app = Flask(__name__)

def generate_frames():
    # 1. Carrega as configurações
    config = AppConfig.from_file("config.yaml")
    
    
    orchestrator = AntiTheftOrchestrator(config)
    
    
    cap = orchestrator.video_source.open()
    metrics = orchestrator.metrics

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            metrics.on_frame()
            timestamp = time.time()

            should_infer = (metrics.frame_count % metrics.frame_skip) == 0
            if should_infer:
                # 1. Deteção
                detection = orchestrator.detector.detect(frame)
                keypoints, scores = [], []
                if isinstance(detection, tuple) and len(detection) == 2:
                    if detection[0] is not None:
                        keypoints = list(detection[0])
                    if detection[1] is not None:
                        scores = list(detection[1])
                
                orchestrator.last_detection = (keypoints, scores)
            elif not orchestrator.cache_result:
                orchestrator.last_detection = ([], [])

            keypoints, scores = orchestrator.last_detection

            # 2. Atividades e Alertas
            alert_text = None
            if keypoints:
                for activity in orchestrator.activities:
                    event = activity.detecta(keypoints, scores, metrics.frame_count, timestamp)
                    if event:
                        orchestrator.alert_dispatcher.dispatch(event)
                        alert_text = f"ALERTA: {event.tipo} ({event.confianca:.0%})"

            # 3. Renderização do Esqueleto
            output = orchestrator.renderer.render(frame, keypoints, scores)
            
            # 4. Enviar para a web
            ret, buffer = cv2.imencode('.jpg', output)
            if not ret:
                break
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()

@app.route('/')
def index():
    return """
    <html>
        <head><title>Sistema Anti-Furto</title></head>
        <body style="text-align: center; background-color: #333; color: white;">
            <h1>Sistema Anti-Furto - Monitorização em Tempo Real</h1>
            <img src="/video_feed" width="800" style="border: 2px solid white;" />
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("[INFO] Servidor Web a iniciar na porta 5000...")
    app.run(host='0.0.0.0', port=5000, debug=False)