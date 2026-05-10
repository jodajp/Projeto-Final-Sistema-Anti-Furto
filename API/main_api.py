from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import glob

app = FastAPI(title="Corner Anti-Theft API", version="1.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringe-se isto ao domínio do Dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminho para a pasta onde o alert_system.py guarda os JSONs
# Assumindo que a API é corrida a partir da raiz do projeto:
ALERTAS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Alertas')

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