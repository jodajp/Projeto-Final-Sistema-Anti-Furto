import sqlite3
import os
import threading
import time
import requests

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
# Recua uma pasta para a raiz do Edge
PASTA_EDGE = os.path.dirname(DIR_ATUAL) 
DB_PATH = os.path.join(PASTA_EDGE, "alertas_oficial.db")

# IP DA TUA NUVEM
CLOUD_API_URL = "http://20.251.152.37:8000/api/alertas/sincronizar"

class DatabaseHandler:
    def __init__(self):
        # Conexão principal para a câmara (escrita rápida)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Criação da tabela de buffer local
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS alertas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                tipo_alerta TEXT,
                confianca REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sincronizado INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

        # ===============================================================
        # NOVO: Inicia a Sincronização em Background Automática
        # daemon=True significa que a thread morre quando fechares a câmara
        # ===============================================================
        self.sync_thread = threading.Thread(target=self._motor_de_sincronizacao, daemon=True)
        self.sync_thread.start()

    def salvar_alerta(self, track_id, tipo_alerta, confianca):
        """ Guarda o alerta instantaneamente no Edge (offline) """
        try:
            self.cursor.execute("""
                INSERT INTO alertas (track_id, tipo_alerta, confianca, sincronizado) 
                VALUES (?, ?, ?, 0)
            """, (track_id, tipo_alerta, confianca))
            self.conn.commit()
        except Exception as e:
            print(f"[Edge DB] Erro a guardar localmente: {e}")

    def _motor_de_sincronizacao(self):
        """ 
        Corre em paralelo (Background). Lê do SQLite e envia para a Nuvem.
        Como corre noutra Thread, NUNCA atrasa os FPS da câmara.
        """
        print("[Edge Sync] Motor de Sincronização Cloud iniciado em background.")
        while True:
            try:
                # Usa uma ligação separada para a Thread para evitar "Database Locks" no SQLite
                sync_conn = sqlite3.connect(DB_PATH)
                sync_cursor = sync_conn.cursor()
                
                # Pede apenas o que não foi sincronizado
                sync_cursor.execute("SELECT id, track_id, tipo_alerta, confianca, timestamp FROM alertas WHERE sincronizado = 0")
                pendentes = sync_cursor.fetchall()
                
                for alerta in pendentes:
                    db_id, track_id, tipo_alerta, confianca, ts = alerta
                    payload = {"track_id": track_id, "tipo_alerta": tipo_alerta, "confianca": confianca, "timestamp": ts}
                    
                    # Envia para a nuvem
                    resp = requests.post(CLOUD_API_URL, json=payload, timeout=3.0)
                    
                    if resp.status_code == 200:
                        sync_cursor.execute("UPDATE alertas SET sincronizado = 1 WHERE id = ?", (db_id,))
                        sync_conn.commit()
                        print(f" -> [Cloud] Alerta {db_id} sincronizado com sucesso!")
                
                sync_conn.close()
            except requests.exceptions.RequestException:
                # Sem Internet: Ignora silenciosamente, os dados estão salvos no disco
                pass
            except Exception as e:
                pass
            
            # Aguarda 5 segundos antes de verificar novamente
            time.sleep(5)