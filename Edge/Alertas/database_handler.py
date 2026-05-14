import sqlite3
import time

class DatabaseHandler:
    def __init__(self, db_name="Edge/Alertas/database_manager.db"):
        """Inicializa a ligação e garante que a tabela existe."""
        self.db_name = db_name
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        """Cria a tabela de alertas se ela não existir."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alertas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                track_id INTEGER,
                tipo_alerta TEXT,
                confianca REAL
            )
        ''')
        self.conn.commit()

    def salvar_alerta(self, track_id, tipo_alerta, confianca):
        """Insere um novo alerta na base de dados."""
        try:
            timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO alertas (timestamp, track_id, tipo_alerta, confianca)
                VALUES (?, ?, ?, ?)
            ''', (timestamp_str, track_id, tipo_alerta, round(confianca, 2)))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[ERRO DB] Falha ao salvar: {e}")
            return False

    def fechar(self):
        """Fecha a ligação com segurança."""
        self.conn.close()