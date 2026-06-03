import sqlite3
import requests
import time
from datetime import datetime
import os

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
# CORREÇÃO: Apontar para o ficheiro correto na mesma diretoria do Edge
DB_PATH = os.path.join(DIR_ATUAL, "alertas_oficial.db")

# Substituir localhost pelo IP da máquina se a cloud estiver noutro PC
API_URL = "http://localhost:8000/api/alertas/sincronizar"

def iniciar_dispatcher():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DISPATCHER] Iniciado. Estratégia Store-and-Forward ativa...")

    while True:
        conn = None
        try:
            # Abrir ligação, ler/escrever e fechar logo a seguir para evitar DB Locks
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row  
            cursor = conn.cursor()

            # 1. Procurar APENAS alertas que não chegaram à Cloud
            cursor.execute("SELECT * FROM alertas WHERE sincronizado = 0")
            alertas_pendentes = cursor.fetchall()

            if alertas_pendentes:
                print(f"[DISPATCHER] {len(alertas_pendentes)} alertas locais pendentes. A tentar envio para o cluster...")

                for alerta in alertas_pendentes:
                    payload = {
                        "track_id": alerta['track_id'],
                        "tipo_alerta": alerta['tipo_alerta'],
                        "confianca": alerta['confianca'],
                        "timestamp": alerta['timestamp']
                    }

                    # 3 segundos é o limite. Se a API não responder, assume falha de rede e desiste.
                    resposta = requests.post(API_URL, json=payload, timeout=3)

                    if resposta.status_code == 200:
                        cursor.execute("UPDATE alertas SET sincronizado = 1 WHERE id = ?", (alerta['id'],))
                        conn.commit()
                        print(f"[DISPATCHER] Sucesso: Alerta #{alerta['id']} transferido para Postgres.")
                    else:
                        print(f"[DISPATCHER] API rejeitou (HTTP {resposta.status_code}). Abortando lote.")
                        break

        except requests.exceptions.RequestException:
            print("[DISPATCHER] ⚠️ Falha na Rede Edge-Cloud. A reter os dados no cofre local...")
        except sqlite3.OperationalError:
            print("[DISPATCHER] A aguardar criação do schema pelo módulo de Visão Computacional...")
        except Exception as e:
            print(f"[DISPATCHER] Erro crítico: {e}")
        finally:
            if conn:
                conn.close()

        # Janela de respiro para o CPU
        time.sleep(5)

if __name__ == "__main__":
    iniciar_dispatcher()