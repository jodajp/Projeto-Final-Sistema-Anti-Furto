import sqlite3
import os
import threading
import time
import requests
from datetime import datetime

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_EDGE = os.path.dirname(DIR_ATUAL) 
DB_PATH = os.path.join(PASTA_EDGE, "dados_oficial.db")


class DatabaseHandler:
    def __init__(self, api_base_url="http://projeto-antifurto-vm1.norwayeast.cloudapp.azure.com:8000"):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.url_alertas = f"{api_base_url}/api/alertas/sincronizar"
        self.url_metricas = f"{api_base_url}/api/metricas/registar"
        self.url_zonas = f"{api_base_url}/api/zonas/sincronizar"
        
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
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                zone_id INTEGER,
                zone_name TEXT,
                hand TEXT,
                deceleration_ratio REAL,
                arm_flex_ratio REAL,
                arm_length REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                sincronizado INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

        # Inicia o motor de sincronização
        self.sync_thread = threading.Thread(target=self._motor_de_sincronizacao, daemon=True)
        self.sync_thread.start()

    # ================= ESCRITA LOCAL (OFFLINE) =================

    # ================= ESCRITA LOCAL (OFFLINE) =================
    
    def salvar_alerta(self, track_id, tipo_alerta, confianca):
        """ Guarda o alerta instantaneamente no Edge """
        try:
            self.cursor.execute("""
                INSERT INTO alertas (track_id, tipo_alerta, confianca, sincronizado) 
                VALUES (?, ?, ?, 0)
            """, (
                int(track_id) if track_id is not None else None, 
                str(tipo_alerta), 
                float(confianca) if confianca is not None else 0.0
            ))
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
                str(metricas_data.get("node_id", "unknown_node")),
                float(metricas_data.get("fps", 0.0)),
                int(metricas_data.get("frame_count", 0)),
                int(metricas_data.get("detection_count", 0)),
                int(metricas_data.get("inference_calls", 0)),
                float(metricas_data.get("average_inference_ms", 0.0)),
                float(metricas_data.get("success_rate", 0.0)),
                float(metricas_data.get("uptime_seconds", 0.0)),
                int(metricas_data.get("pessoas_detetadas", 0))
            ))
            self.conn.commit()
        except Exception as e:
            print(f"[Edge DB] Erro a guardar métrica local: {e}")

    def salvar_evento_zona(self, track_id, zone_id, zone_name, hand, deceleration_ratio, arm_flex_ratio, arm_length, timestamp=None):
        """Guarda um evento de zona (grab) no Edge."""
        try:
            self.cursor.execute("""
                INSERT INTO zones (
                    track_id, zone_id, zone_name, hand, deceleration_ratio, arm_flex_ratio, arm_length, timestamp, sincronizado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), 0)
            """, (
                int(track_id) if track_id is not None else None, 
                int(zone_id) if zone_id is not None else None, 
                str(zone_name), 
                str(hand), 
                float(deceleration_ratio) if deceleration_ratio is not None else 0.0, 
                float(arm_flex_ratio) if arm_flex_ratio is not None else 0.0, 
                float(arm_length) if arm_length is not None else 0.0, 
                timestamp
            ))
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
        sync_conn = None
        try:
            import struct
            sync_conn = sqlite3.connect(DB_PATH)
            sync_cursor = sync_conn.cursor()
            
            # Helper function to decode bytes safely
            def safe_decode(val):
                if isinstance(val, bytes):
                    return val.decode('utf-8', errors='ignore')
                return val

            def safe_float(val):
                if isinstance(val, bytes):
                    if len(val) == 4:
                        return float(struct.unpack('<f', val)[0])
                    elif len(val) == 8:
                        return float(struct.unpack('<d', val)[0])
                    try:
                        return float(val.decode('utf-8'))
                    except Exception:
                        return 0.0
                return float(val) if val is not None else 0.0

            def safe_int(val):
                if isinstance(val, bytes):
                    if len(val) == 4:
                        return int(struct.unpack('<i', val)[0])
                    elif len(val) == 8:
                        return int(struct.unpack('<q', val)[0])
                    try:
                        return int(val.decode('utf-8'))
                    except Exception:
                        return 0
                return int(val) if val is not None else 0
            
            # 1. ALERTAS
            try:
                sync_cursor.execute("SELECT id, track_id, tipo_alerta, confianca, timestamp FROM alertas WHERE sincronizado = 0")
                for db_id, track_id, tipo_alerta, confianca, ts in sync_cursor.fetchall():
                    payload = {
                        "track_id": safe_int(track_id), 
                        "tipo_alerta": safe_decode(tipo_alerta), 
                        "confianca": safe_float(confianca), 
                        "timestamp": safe_decode(ts)
                    }
                    resp = requests.post(self.url_alertas, json=payload, timeout=3.0)
                    if resp.status_code == 200:
                        sync_cursor.execute("UPDATE alertas SET sincronizado = 1 WHERE id = ?", (db_id,))
                        sync_conn.commit()
                        print(f" -> [Cloud] Alerta {db_id} sincronizado!")
            except requests.exceptions.RequestException:
                pass # Sem rede para alertas, tenta na próxima ronda

            # 2. MÉTRICAS (Agora extrai e envia as pessoas_detetadas)
            try:
                sync_cursor.execute("""
                    SELECT id, node_id, fps, frame_count, detection_count, 
                           inference_calls, average_inference_ms, success_rate, 
                           uptime_seconds, pessoas_detetadas 
                    FROM metricas WHERE sincronizado = 0
                """)
                for metrica in sync_cursor.fetchall():
                    db_id, n_id, fps, f_count, d_count, i_calls, avg_inf, succ, up, pessoas = metrica
                    
                    payload_metrica = {
                        "node_id": safe_decode(n_id),
                        "fps": safe_float(fps),
                        "frame_count": safe_int(f_count),
                        "detection_count": safe_int(d_count),
                        "inference_calls": safe_int(i_calls),
                        "average_inference_ms": safe_float(avg_inf),
                        "success_rate": safe_float(succ),
                        "uptime_seconds": safe_float(up),
                        "pessoas_detetadas": safe_int(pessoas)
                    }
                    
                    resp = requests.post(self.url_metricas, json=payload_metrica, timeout=3.0)
                    if resp.status_code == 200:
                        sync_cursor.execute("UPDATE metricas SET sincronizado = 1 WHERE id = ?", (db_id,))
                        sync_conn.commit()
                        print(f" -> [Cloud] Métrica {db_id} sincronizada!")
            except requests.exceptions.RequestException:
                pass # Sem rede para métricas, tenta na próxima ronda

            # 3. EVENTOS DE ZONES
            try:
                sync_cursor.execute("""
                    SELECT id, track_id, zone_id, zone_name, hand, deceleration_ratio, arm_flex_ratio, arm_length, timestamp 
                    FROM zones WHERE sincronizado = 0
                """)
                grabs = sync_cursor.fetchall()
                if grabs:
                    print(f"[Sync] {len(grabs)} eventos de Zones pendentes para sincronizar")
                    
                for zone_event in grabs:
                    db_id, track_id, zone_id, zone_name, hand, decel, flex, arm_len, ts = zone_event
                    
                    try:
                        # Tenta converter o número "feio" para uma data ISO "bonita" que a Cloud aceita
                        clean_ts = safe_decode(ts)
                        try:
                            ts_iso = datetime.fromtimestamp(float(clean_ts)).isoformat()
                        except (ValueError, TypeError):
                            ts_iso = str(clean_ts) # Fallback de segurança

                        payload_zone = {
                            "track_id": int(safe_int(track_id)),
                            "zone_id": int(safe_int(zone_id)),
                            "zone_name": str(safe_decode(zone_name)),
                            "hand": str(safe_decode(hand)),
                            "deceleration_ratio": float(safe_float(decel)),
                            "arm_flex_ratio": float(safe_float(flex)),
                            "arm_length": float(safe_float(arm_len)),
                            "timestamp": ts_iso
                        }
                        
                        resp = requests.post(self.url_zonas, json=payload_zone, timeout=3.0)
                        
                        if resp.status_code == 200:
                            sync_cursor.execute("UPDATE zones SET sincronizado = 1 WHERE id = ?", (db_id,))
                            sync_conn.commit()
                            print(f" -> [Cloud] Evento Zone {db_id} sincronizado!")
                        else:
                            print(f" [!] Cloud rejeitou Zone {db_id}. Status: {resp.status_code} | Detalhe: {resp.text}")
                    except (ValueError, TypeError) as e:
                        print(f" [!] Erro ao serializar Evento Zone {db_id}: {e} - marcando como sincronizado")
                        sync_cursor.execute("UPDATE zones SET sincronizado = 1 WHERE id = ?", (db_id,))
                        sync_conn.commit()
            except requests.exceptions.RequestException:
                pass # Sem rede para zonas, tenta na próxima ronda
                
        except Exception as e:
            import traceback
            print(f"[Edge Sync] Erro no loop: {e}")
            traceback.print_exc()
        finally:
            if sync_conn:
                try:
                    sync_conn.close()
                except Exception:
                    pass

    def close(self):
        """ Força sincronização de dados pendentes antes de fechar a conexão """
        self._sincronizar_dados()
        try:
            self.conn.close()
        except Exception:
            pass