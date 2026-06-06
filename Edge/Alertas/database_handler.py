import sqlite3
import os
import threading
import time
import requests

# ==========================================
# 1. CONFIGURAÇÕES E PATHS
# ==========================================
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_EDGE = os.path.dirname(DIR_ATUAL) 
DB_PATH = os.path.join(PASTA_EDGE, "dados_oficial.db")


CLOUD_API_ALERTAS = os.getenv("CLOUD_API_ALERTAS", "http://20.251.152.37:8000/api/alertas/sincronizar")
CLOUD_API_METRICAS = os.getenv("CLOUD_API_METRICAS", "http://20.251.152.37:8000/api/metricas/registar")

class DatabaseHandler:
    def __init__(self):
        # Ligação principal (Thread de Vídeo)
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._criar_tabelas()

        # Inicia o motor de sincronização (Thread de Background)
        self.sync_thread = threading.Thread(target=self._motor_de_sincronizacao, daemon=True)
        self.sync_thread.start()

    def _criar_tabelas(self):
        """Inicializa as tabelas locais se não existirem."""
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
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS metricas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                fps REAL,
                frame_count INTEGER,
                detection_count INTEGER,
                inference_calls INTEGER,
                average_inference_ms REAL,
                success_rate REAL,
                uptime_seconds REAL,
                pessoas_detetadas INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sincronizado INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    # ==========================================
    # 2. ESCRITA LOCAL (CÂMARA -> SQLITE)
    # ==========================================
    def salvar_alerta(self, track_id, tipo_alerta, confianca):
        try:
            self.cursor.execute("""
                INSERT INTO alertas (track_id, tipo_alerta, confianca, sincronizado) 
                VALUES (?, ?, ?, 0)
            """, (track_id, tipo_alerta, confianca))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ [Edge DB] Erro a guardar alerta local: {e}")

    def salvar_metrica(self, metricas_data):
        try:
            self.cursor.execute("""
                INSERT INTO metricas (
                    node_id, fps, frame_count, detection_count, 
                    inference_calls, average_inference_ms, success_rate, 
                    uptime_seconds, pessoas_detetadas, sincronizado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                metricas_data.get("node_id", "unknown_node"),
                metricas_data.get("fps", 0.0),
                metricas_data.get("frame_count", 0),
                metricas_data.get("detection_count", 0),
                metricas_data.get("inference_calls", 0),
                metricas_data.get("average_inference_ms", 0.0),
                metricas_data.get("success_rate", 0.0),
                metricas_data.get("uptime_seconds", 0.0),
                metricas_data.get("pessoas_detetadas", 0)
            ))
            self.conn.commit()
        except Exception as e:
            print(f"⚠️ [Edge DB] Erro a guardar métrica local: {e}")

    # ==========================================
    # 3. SINCRONIZAÇÃO (SQLITE -> CLOUD)
    # ==========================================
    def _motor_de_sincronizacao(self):
        print("🚀 [Edge Sync] Motor de Sincronização iniciado.")
        
        # OTIMIZAÇÃO: Usa requests.Session() para reaproveitar a ligação TCP
        # Reduz drasticamente o overhead de rede e tempo de CPU
        http_session = requests.Session()

        while True:
            try:
                # O bloco 'with' garante fecho seguro da BD mesmo que haja crashes na rede
                with sqlite3.connect(DB_PATH) as sync_conn:
                    sync_cursor = sync_conn.cursor()
                    
                    # --- 1. SINCRONIZAR ALERTAS ---
                    sync_cursor.execute("SELECT id, track_id, tipo_alerta, confianca, timestamp FROM alertas WHERE sincronizado = 0")
                    alertas_pendentes = sync_cursor.fetchall()
                    
                    for db_id, track_id, tipo_alerta, confianca, ts in alertas_pendentes:
                        payload = {"track_id": track_id, "tipo_alerta": tipo_alerta, "confianca": confianca, "timestamp": ts}
                        resp = http_session.post(CLOUD_API_ALERTAS, json=payload, timeout=3.0)
                        
                        if resp.status_code == 200:
                            sync_cursor.execute("UPDATE alertas SET sincronizado = 1 WHERE id = ?", (db_id,))
                            sync_conn.commit()
                            print(f"✅ [Cloud] Alerta {db_id} sincronizado!")

                    # --- 2. SINCRONIZAR MÉTRICAS ---
                    sync_cursor.execute("""
                        SELECT id, node_id, fps, frame_count, detection_count, 
                               inference_calls, average_inference_ms, success_rate, 
                               uptime_seconds, pessoas_detetadas 
                        FROM metricas WHERE sincronizado = 0
                    """)
                    metricas_pendentes = sync_cursor.fetchall()
                    
                    for metrica in metricas_pendentes:
                        db_id, n_id, fps, f_count, d_count, i_calls, avg_inf, succ, up, pessoas = metrica
                        payload_metrica = {
                            "node_id": n_id, "fps": fps, "frame_count": f_count, 
                            "detection_count": d_count, "inference_calls": i_calls, 
                            "average_inference_ms": avg_inf, "success_rate": succ, 
                            "uptime_seconds": up, "pessoas_detetadas": pessoas
                        }
                        
                        resp = http_session.post(CLOUD_API_METRICAS, json=payload_metrica, timeout=3.0)
                        
                        if resp.status_code == 200:
                            sync_cursor.execute("UPDATE metricas SET sincronizado = 1 WHERE id = ?", (db_id,))
                            sync_conn.commit()
                            print(f"📊 [Cloud] Métrica {db_id} sincronizada!")
                            
            except requests.exceptions.RequestException:
                pass # Silencioso: Sem internet, os dados esperam seguros no SQLite
            except Exception as e:
                print(f"❌ [Edge Sync] Erro interno: {e}")
            
            time.sleep(5)