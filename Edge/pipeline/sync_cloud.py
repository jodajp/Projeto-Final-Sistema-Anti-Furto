import sqlite3
import requests
import time
import os

# IP DA TUA NUVEM AQUI!
CLOUD_API_URL = "http://20.251.152.37:8000/api/alertas/sincronizar"

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(DIR_ATUAL), "alertas_oficial.db")

def sincronizar_alertas():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Pega nos alertas que ainda não foram para a cloud
        cursor.execute("SELECT id, track_id, tipo_alerta, confianca, timestamp FROM alertas WHERE sincronizado = 0")
        pendentes = cursor.fetchall()
        
        if pendentes:
            print(f"[{time.strftime('%H:%M:%S')}] A enviar {len(pendentes)} alertas offline para a Nuvem...")

        for alerta in pendentes:
            db_id, track_id, tipo_alerta, confianca, ts = alerta
            payload = {"track_id": track_id, "tipo_alerta": tipo_alerta, "confianca": confianca, "timestamp": ts}
            
            # Envia
            resp = requests.post(CLOUD_API_URL, json=payload, timeout=5.0)
            
            if resp.status_code == 200:
                cursor.execute("UPDATE alertas SET sincronizado = 1 WHERE id = ?", (db_id,))
                conn.commit()
                
    except Exception as e:
        pass # Se falhar (sem net), não faz mal, tenta na próxima volta
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("=== Ponte Edge -> Cloud Ativada ===")
    while True:
        sincronizar_alertas()
        time.sleep(5) # Tenta enviar de 5 em 5 segundos