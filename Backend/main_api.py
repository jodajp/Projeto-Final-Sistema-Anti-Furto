from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import glob
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Corner Anti-Theft API", version="1.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringe-se isto ao domínio do Dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos para as pastas de dados
# Assumindo que a API é corrida a partir da raiz do projeto:
ALERTAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Alertas')
METRICAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Metricas')

# Criar diretório de métricas se não existir
os.makedirs(METRICAS_DIR, exist_ok=True)


# Modelos Pydantic para respostas
class MetricaNode(BaseModel):
    node_id: str
    timestamp: float
    fps: float
    frame_count: int
    detection_count: int
    inference_calls: int
    average_inference_ms: float
    success_rate: float
    uptime_seconds: float
    
class MetricasCluster(BaseModel):
    timestamp: datetime
    nodes: List[MetricaNode]
    media_fps: float
    total_detections: int
    total_frames: int

@app.get("/")
def read_root():
    return {"status": "A Corner API está a correr!"}

@app.get("/api/alertas/recentes")
def get_recent_alerts():
    """
    Vai à pasta Alertas, encontra o ficheiro JSON mais recente e devolve o seu conteúdo.
    Se não houver ficheiros, devolve uma lista vazia.
    """
    if not os.path.exists(ALERTAS_DIR):
        return {"alertas": []}

    # Procurar todos os ficheiros JSON na pasta Alertas
    ficheiros_json = glob.glob(os.path.join(ALERTAS_DIR, "eventos_*.json"))
    
    if not ficheiros_json:
        return {"alertas": []}

    # Ordenar por data de modificação (o mais recente primeiro)
    ficheiro_mais_recente = max(ficheiros_json, key=os.path.getmtime)
    
    try:
        with open(ficheiro_mais_recente, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            
            # Ordenar os alertas dentro do ficheiro para garantir que o mais novo aparece em cima
            # (Assumindo que os dados são uma lista de dicionários com 'timestamp')
            if isinstance(dados, list):
                 # Ordena de forma descendente usando o timestamp
                 dados_ordenados = sorted(dados, key=lambda x: x.get('timestamp', 0), reverse=True)
                 # Devolve apenas os 10 mais recentes para não sobrecarregar o Dashboard
                 return {"alertas": dados_ordenados[:10]}
            
            return {"alertas": dados}
            
    except Exception as e:
        return {"erro": f"Falha ao ler o ficheiro: {str(e)}", "alertas": []}


# ============ ENDPOINTS DE MÉTRICAS ============

@app.get("/api/metricas/atuais")
def get_current_metrics():
    """
    Obtém as métricas atuais de todos os nós do cluster.
    Procura os ficheiros mais recentes de cada nó na pasta Metricas.
    """
    if not os.path.exists(METRICAS_DIR):
        return {"metricas": [], "total_nodes": 0}
    
    # Procurar ficheiros de métricas: metricas_node_*.json
    ficheiros_metricas = glob.glob(os.path.join(METRICAS_DIR, "metricas_*.json"))
    
    if not ficheiros_metricas:
        return {
            "metricas": [],
            "total_nodes": 0,
            "mensagem": "Nenhuma métrica disponível. Os nós Edge precisam de guardar ficheiros de métricas."
        }
    
    metricas_por_node = {}
    
    # Agrupar por node_id e pegar o ficheiro mais recente de cada um
    for ficheiro in ficheiros_metricas:
        try:
            nome = os.path.basename(ficheiro)  # ex: metricas_node1_20230515_143022.json
            parts = nome.replace("metricas_", "").replace(".json", "").split("_")
            if len(parts) >= 3:
                node_id = "_".join(parts[:-2])  # Pega tudo exceto data e hora
                
                if node_id not in metricas_por_node or os.path.getmtime(ficheiro) > os.path.getmtime(metricas_por_node[node_id]):
                    metricas_por_node[node_id] = ficheiro
        except:
            continue
    
    # Ler as métricas mais recentes de cada nó
    metricas_atuais = []
    for node_id, ficheiro in metricas_por_node.items():
        try:
            with open(ficheiro, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                dados['node_id'] = node_id
                metricas_atuais.append(dados)
        except Exception as e:
            print(f"Erro ao ler métricas de {node_id}: {str(e)}")
    
    return {
        "metricas": metricas_atuais,
        "total_nodes": len(metricas_atuais),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metricas/cluster")
def get_cluster_metrics():
    """
    Calcula e retorna métricas agregadas do cluster inteiro.
    """
    resultado_atuais = get_current_metrics()
    metricas = resultado_atuais.get("metricas", [])
    
    if not metricas:
        return {
            "cluster_metrics": None,
            "mensagem": "Não há métricas disponíveis"
        }
    
    # Calcular agregações
    total_fps = sum(m.get('fps', 0) for m in metricas)
    total_frames = sum(m.get('frame_count', 0) for m in metricas)
    total_detections = sum(m.get('detection_count', 0) for m in metricas)
    total_inferences = sum(m.get('inference_calls', 0) for m in metricas)
    
    media_fps = total_fps / len(metricas) if metricas else 0
    
    # Calcular uptime máximo (do nó com mais tempo de funcionamento)
    max_uptime = max((m.get('uptime_seconds', 0) for m in metricas), default=0)
    
    # Tempo médio de inferência
    tempo_medio_inferencia = sum(m.get('average_inference_ms', 0) for m in metricas) / len(metricas) if metricas else 0
    
    # Taxa de sucesso média
    taxa_sucesso_media = sum(m.get('success_rate', 0) for m in metricas) / len(metricas) if metricas else 0
    
    return {
        "cluster_metrics": {
            "num_nodes": len(metricas),
            "media_fps": round(media_fps, 2),
            "total_frames": total_frames,
            "total_detections": total_detections,
            "total_inference_calls": total_inferences,
            "tempo_medio_inferencia_ms": round(tempo_medio_inferencia, 2),
            "taxa_sucesso_media_pct": round(taxa_sucesso_media, 2),
            "uptime_maximo_segundos": int(max_uptime)
        },
        "nodes": metricas,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/metricas/node/{node_id}")
def get_node_metrics(node_id: str):
    """
    Obtém métricas de um nó específico.
    """
    if not os.path.exists(METRICAS_DIR):
        return {"erro": f"Nó {node_id} não encontrado", "metricas": None}
    
    # Procurar ficheiro mais recente do nó
    ficheiros_node = glob.glob(os.path.join(METRICAS_DIR, f"metricas_{node_id}_*.json"))
    
    if not ficheiros_node:
        return {"erro": f"Nó {node_id} não encontrado", "metricas": None}
    
    ficheiro_mais_recente = max(ficheiros_node, key=os.path.getmtime)
    
    try:
        with open(ficheiro_mais_recente, 'r', encoding='utf-8') as f:
            dados = json.load(f)
            dados['node_id'] = node_id
            return {
                "node_id": node_id,
                "metricas": dados,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {"erro": f"Erro ao ler métricas de {node_id}: {str(e)}", "metricas": None}


@app.get("/api/metricas/historico")
def get_metrics_history(node_id: Optional[str] = None, limite: int = 50):
    """
    Obtém o histórico de métricas.
    Se node_id for especificado, retorna apenas o histórico desse nó.
    O parâmetro 'limite' controla quantos registos retornar (máximo 100).
    """
    if not os.path.exists(METRICAS_DIR):
        return {"historico": [], "total": 0}
    
    limite = min(limite, 100)  # Máximo de 100 registos
    
    # Procurar ficheiros de métricas
    if node_id:
        ficheiros = glob.glob(os.path.join(METRICAS_DIR, f"metricas_{node_id}_*.json"))
    else:
        ficheiros = glob.glob(os.path.join(METRICAS_DIR, "metricas_*.json"))
    
    # Ordenar por data de modificação (mais recente primeiro)
    ficheiros.sort(key=os.path.getmtime, reverse=True)
    
    historico = []
    for ficheiro in ficheiros[:limite]:
        try:
            with open(ficheiro, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                nome = os.path.basename(ficheiro)
                parts = nome.replace("metricas_", "").replace(".json", "").split("_")
                if len(parts) >= 3:
                    dados['node_id'] = "_".join(parts[:-2])
                    historico.append(dados)
        except:
            continue
    
    return {
        "historico": historico,
        "total": len(historico),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/metricas/registar")
def register_metrics(metricas_data: Dict[str, Any]):
    """
    Endpoint para registar novas métricas do Edge.
    Espera um JSON com a estrutura das métricas.
    
    Exemplo de payload:
    {
        "node_id": "node1",
        "fps": 25.5,
        "frame_count": 1500,
        "detection_count": 45,
        "inference_calls": 750,
        "average_inference_ms": 15.3,
        "success_rate": 6.0,
        "uptime_seconds": 3600
    }
    """
    try:
        node_id = metricas_data.get('node_id', 'unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Guardar ficheiro com timestamp
        ficheiro = os.path.join(METRICAS_DIR, f"metricas_{node_id}_{timestamp}.json")
        
        with open(ficheiro, 'w', encoding='utf-8') as f:
            json.dump(metricas_data, f, indent=2)
        
        return {
            "status": "sucesso",
            "mensagem": f"Métricas registadas para {node_id}",
            "ficheiro": os.path.basename(ficheiro)
        }
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": f"Erro ao registar métricas: {str(e)}"
        }