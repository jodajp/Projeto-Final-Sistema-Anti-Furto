from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from sqlalchemy import func
from datetime import datetime
import os
import json
import glob
from pathlib import Path

# Importações dos teus módulos organizados
from app.database import engine, Base, get_db
from app.models.metrica import MetricaNodeModel
from app.schemas.metrica import MetricaNodeCreate, MetricasClusterResponse, ClusterMetricsSummary

# Cria as tabelas no PostgreSQL se elas não existirem no arranque
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Corner Anti-Theft Enterprise API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIGURAÇÃO DE CAMINHOS (Ajustados para a nova estrutura Backend/app/) ===
# Como o ficheiro está em Backend/app/main.py, subimos 3 níveis para chegar à raiz do projeto
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ALERTAS_DIR = os.path.join(BASE_DIR, 'Alertas')
METRICAS_DIR = os.path.join(BASE_DIR, 'Metricas')  # Usado para o last_frame.jpg

# Garantir que os diretórios existem
os.makedirs(ALERTAS_DIR, exist_ok=True)
os.makedirs(METRICAS_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "A Corner Enterprise API está a correr a 100% com PostgreSQL!"}


# ============ ENDPOINTS DE ALERTAS (Mantidos por Ficheiro) ============

@app.get("/api/alertas/recentes")
def get_recent_alerts():
    """Vai à pasta Alertas, encontra o ficheiro JSON mais recente e devolve o seu conteúdo."""
    if not os.path.exists(ALERTAS_DIR):
        return {"alertas": []}

    ficheiros_json = glob.glob(os.path.join(ALERTAS_DIR, "eventos_*.json"))
    if not ficheiros_json:
        return {"alertas": []}

    ficheiro_mais_recente = max(ficheiros_json, key=os.path.getmtime)
    
    try:
        with open(ficheiro_mais_recente, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            if isinstance(dados, list):
                 dados_ordenados = sorted(dados, key=lambda x: x.get('timestamp', 0), reverse=True)
                 return {"alertas": dados_ordenados[:10]}
            return {"alertas": dados}
    except Exception as e:
        return {"erro": f"Falha ao ler o ficheiro de alertas: {str(e)}", "alertas": []}


# ============ ENDPOINTS DE MÉTRICAS (Evoluídos para PostgreSQL) ============

@app.post("/api/metricas/registar")
def register_metrics(metrica: MetricaNodeCreate, db: Session = Depends(get_db)):
    """Recebe os dados em tempo real do Edge e grava-os no PostgreSQL."""
    try:
        nova_metrica = MetricaNodeModel(
            node_id=metrica.node_id,
            timestamp=metrica.timestamp,
            fps=metrica.fps,
            frame_count=metrica.frame_count,
            detection_count=metrica.detection_count,
            inference_calls=metrica.inference_calls,
            average_inference_ms=metrica.average_inference_ms,
            success_rate=metrica.success_rate,
            uptime_seconds=metrica.uptime_seconds
        )
        db.add(nova_metrica)
        db.commit()
        return {
            "status": "sucesso", 
            "mensagem": f"Métricas registadas no PostgreSQL para {metrica.node_id}"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Erro ao registar na BD: {str(e)}")


@app.get("/api/metricas/atuais")
def get_current_metrics(db: Session = Depends(get_db)):
    """Obtém as métricas mais recentes de cada nó ativo diretamente do PostgreSQL."""
    try:
        # Query que encontra o ID mais alto (mais recente) para cada node_id distinto
        subquery = db.query(MetricaNodeModel.node_id, engine.dialect.functions.max(MetricaNodeModel.id).label("max_id")).group_by(MetricaNodeModel.node_id).subquery()
        metricas_atuais = db.query(MetricaNodeModel).join(subquery, MetricaNodeModel.id == subquery.c.max_id).all()
        
        return {
            "metricas": metricas_atuais,
            "total_nodes": len(metricas_atuais),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler métricas atuais: {str(e)}")


@app.get("/api/metricas/cluster", response_model=MetricasClusterResponse)
def get_cluster_metrics(db: Session = Depends(get_db)):
    # 1. Obter as métricas mais recentes (o teu código corrigido)
    subquery = db.query(
        MetricaNodeModel.node_id, 
        func.max(MetricaNodeModel.id).label("max_id")
    ).group_by(MetricaNodeModel.node_id).subquery()

    metricas_atuais = db.query(MetricaNodeModel).join(
        subquery, 
        MetricaNodeModel.id == subquery.c.max_id
    ).all()

    if not metricas_atuais:
        return {
            "cluster_metrics": None,
            "nodes": [],
            "timestamp": datetime.now().isoformat()
        }

    # 2. CALCULAR OS TOTAIS (Necessário para preencher o MetricasClusterResponse)
    total_fps = sum(m.fps for m in metricas_atuais)
    total_frames = sum(m.frame_count for m in metricas_atuais)
    total_detections = sum(m.detection_count for m in metricas_atuais)
    total_inferences = sum(m.inference_calls for m in metricas_atuais)
    media_inferencia = sum(m.average_inference_ms for m in metricas_atuais) / len(metricas_atuais)
    media_sucesso = sum(m.success_rate for m in metricas_atuais) / len(metricas_atuais)

    summary = ClusterMetricsSummary(
        num_nodes=len(metricas_atuais),
        media_fps=round(total_fps / len(metricas_atuais), 2),
        total_frames=total_frames,
        total_detections=total_detections,
        total_inference_calls=total_inferences,
        tempo_medio_inferencia_ms=round(media_inferencia, 2),
        taxa_sucesso_media_pct=round(media_sucesso, 2),
        uptime_maximo_segundos=int(max(m.uptime_seconds for m in metricas_atuais))
    )

    # 3. Retornar no formato que o Pydantic espera
    return {
        "cluster_metrics": summary,
        "nodes": metricas_atuais,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metricas/node/{node_id}")
def get_node_metrics(node_id: str, db: Session = Depends(get_db)):
    """Obtém a métrica mais recente de um nó específico a partir do PostgreSQL."""
    metrica = db.query(MetricaNodeModel).filter(MetricaNodeModel.node_id == node_id).order_by(MetricaNodeModel.id.desc()).first()
    
    if not metrica:
        raise HTTPException(status_code=404, detail=f"Nó {node_id} não encontrado na Base de Dados.")
        
    return {
        "node_id": node_id,
        "metricas": metrica,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metricas/historico")
def get_metrics_history(node_id: Optional[str] = None, limite: int = 50, db: Session = Depends(get_db)):
    """Obtém o histórico de todas as métricas gravadas na BD (com limite adaptável)."""
    limite = min(limite, 100)  # Proteção para não sobrecarregar a rede
    
    query = db.query(MetricaNodeModel)
    if node_id:
        query = query.filter(MetricaNodeModel.node_id == node_id)
        
    historico = query.order_by(MetricaNodeModel.id.desc()).limit(limite).all()
    
    return {
        "historico": historico,
        "total": len(historico),
        "timestamp": datetime.now().isoformat()
    }


# ============ ENDPOINTS DE VIDEO/STREAM (Mantidos por Ficheiro/Live) ============

@app.get("/api/video/frame")
def get_current_frame():
    """Retorna o último frame capturado em JPEG pelo nó Edge."""
    frame_path = os.path.join(METRICAS_DIR, 'last_frame.jpg')
    
    if not os.path.exists(frame_path):
        return {"erro": "Nenhum frame disponível", "mensagem": "O Edge precisa de estar a correr para gerar frames."}
    
    try:
        return FileResponse(frame_path, media_type="image/jpeg")
    except Exception as e:
        return {"erro": f"Erro ao servir o frame: {str(e)}"}


@app.get("/api/video/stream")
def get_video_stream():
    """Retorna o stream MJPEG contínuo do feed da câmara para o Dashboard."""
    frame_path = os.path.join(METRICAS_DIR, 'last_frame.jpg')
    
    def frame_generator():
        import time
        while True:
            try:
                if os.path.exists(frame_path):
                    with open(frame_path, 'rb') as f:
                        frame_data = f.read()
                    
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                        + frame_data + b'\r\n'
                    )
                time.sleep(0.067)  # Estabiliza o stream a cerca de ~15 FPS
            except Exception as e:
                print(f"Erro no gerador de stream: {str(e)}")
                break
    
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )