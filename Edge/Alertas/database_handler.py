import sqlite3
import os

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
# Recua uma pasta para a raiz do Edge
PASTA_EDGE = os.path.dirname(DIR_ATUAL) 
DB_PATH = os.path.join(PASTA_EDGE, "alertas_oficial.db")

class DatabaseHandler:
    def __init__(self):
        # check_same_thread=False é crucial se o YOLO-Pose correr noutra thread
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

    def salvar_alerta(self, track_id, tipo_alerta, confianca):
        """
        Guarda o alerta ESTRITAMENTE no SQLite local. 
        Garante a resiliência no retalho: se não há internet, o dado não se perde.
        """
        try:
            self.cursor.execute("""
                INSERT INTO alertas (track_id, tipo_alerta, confianca, sincronizado) 
                VALUES (?, ?, ?, 0)
            """, (track_id, tipo_alerta, confianca))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[ERRO SQLITE Edge] Falha ao guardar localmente: {e}")
            return False