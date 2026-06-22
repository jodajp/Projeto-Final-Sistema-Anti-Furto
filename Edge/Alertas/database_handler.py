import sqlite3
import os
import threading
import time
import requests

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_EDGE = os.path.dirname(DIR_ATUAL) 
DB_PATH = os.path.join(PASTA_EDGE, "dados_oficial.db")

# URLs DA TUA NUVEM (API)
CLOUD_API_ALERTAS = "http://20.251.152.37:8000/api/alertas/sincronizar"
CLOUD_API_METRICAS = "http://20.251.152.37:8000/api/metricas/registar"

class DatabaseHandler:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # 1. Tabela local para ALERTAS
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

        # 2. Tabela local para MÉTRICAS (Agora inclui pessoas_detetadas)
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

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS zone_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                zone_id INTEGER,
                zone_name TEXT,
                hand TEXT,
                x REAL,
                y REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sincronizado INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

        # Inicia o motor de sincronização
        self.sync_thread = threading.Thread(target=self._motor_de_sincronizacao, daemon=True)
        self.sync_thread.start()

    # ================= ESCRITA LOCAL (OFFLINE) =================

    def salvar_alerta(self, track_id, tipo_alerta, confianca):
        """ Guarda o alerta instantaneamente no Edge """
        try:
            self.cursor.execute("""
                INSERT INTO alertas (track_id, tipo_alerta, confianca, sincronizado) 
                VALUES (?, ?, ?, 0)
            """, (track_id, tipo_alerta, confianca))
            self.conn.commit()
        except Exception as e:
            print(f"[Edge DB] Erro a guardar alerta local: {e}")

    def salvar_metrica(self, metricas_data):
        """ Guarda as métricas instantaneamente no Edge. Recebe o dicionário do orchestrator.py """
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
            print(f"[Edge DB] Erro a guardar métrica local: {e}")

    def salvar_evento_zona(self, track_id, zone_id, zone_name, hand, x, y, timestamp=None):
        """Guarda um evento de zona do braço no Edge."""
        try:
            self.cursor.execute("""
                INSERT INTO zone_events (
                    track_id, zone_id, zone_name, hand, x, y, timestamp, sincronizado
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), 0)
            """, (track_id, zone_id, zone_name, hand, x, y, timestamp))
            self.conn.commit()
        except Exception as e:
            print(f"[Edge DB] Erro a guardar evento de zona local: {e}")

    # ================= SINCRONIZAÇÃO (CLOUD) =================

    def _motor_de_sincronizacao(self):
        """ Corre em background para não atrasar o vídeo """
        print("[Edge Sync] Motor de Sincronização iniciado.")
        while True:
            self._sincronizar_dados()
            time.sleep(5)

    def _sincronizar_dados(self):
        """ Realiza o envio de alertas e métricas não sincronizados para a nuvem """
        try:
            sync_conn = sqlite3.connect(DB_PATH)
            sync_cursor = sync_conn.cursor()
            
            # 1. ALERTAS
            sync_cursor.execute("SELECT id, track_id, tipo_alerta, confianca, timestamp FROM alertas WHERE sincronizado = 0")
            for db_id, track_id, tipo_alerta, confianca, ts in sync_cursor.fetchall():
                payload = {"track_id": track_id, "tipo_alerta": tipo_alerta, "confianca": confianca, "timestamp": ts}
                resp = requests.post(CLOUD_API_ALERTAS, json=payload, timeout=3.0)
                if resp.status_code == 200:
                    sync_cursor.execute("UPDATE alertas SET sincronizado = 1 WHERE id = ?", (db_id,))
                    sync_conn.commit()
                    print(f" -> [Cloud] Alerta {db_id} sincronizado!")

            # 2. MÉTRICAS (Agora extrai e envia as pessoas_detetadas)
            sync_cursor.execute("""
                SELECT id, node_id, fps, frame_count, detection_count, 
                       inference_calls, average_inference_ms, success_rate, 
                       uptime_seconds, pessoas_detetadas 
                FROM metricas WHERE sincronizado = 0
            """)
            for metrica in sync_cursor.fetchall():
                db_id, n_id, fps, f_count, d_count, i_calls, avg_inf, succ, up, pessoas = metrica
                
                payload_metrica = {
                    "node_id": n_id,
                    "fps": fps,
                    "frame_count": f_count,
                    "detection_count": d_count,
                    "inference_calls": i_calls,
                    "average_inference_ms": avg_inf,
                    "success_rate": succ,
                    "uptime_seconds": up,
                    "pessoas_detetadas": pessoas
                }
                
                resp = requests.post(CLOUD_API_METRICAS, json=payload_metrica, timeout=3.0)
                if resp.status_code == 200:
                    sync_cursor.execute("UPDATE metricas SET sincronizado = 1 WHERE id = ?", (db_id,))
                    sync_conn.commit()
                    print(f" -> [Cloud] Métrica {db_id} sincronizada!")

            sync_conn.close()
            
        except requests.exceptions.RequestException:
            pass # Sem net, tenta na próxima ronda
        except Exception as e:
            print(f"[Edge Sync] Erro no loop: {e}")

    def close(self):
        """ Força sincronização de dados pendentes antes de fechar a conexão """
        self._sincronizar_dados()
        try:
            self.conn.close()
        except Exception:
            pass