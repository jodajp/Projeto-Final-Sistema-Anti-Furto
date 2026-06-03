import sqlite3
import os

db_path = os.path.join("Edge", "Alertas", "alertas_oficial.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Injetar 3 alertas simulados
for i in range(3):
    cursor.execute("""
        INSERT INTO alertas (track_id, tipo_alerta, confianca, sincronizado) 
        VALUES (?, ?, ?, 0)
    """, (i+100, 'Movimento Suspeito - Simulação', 0.95))

conn.commit()
conn.close()
print("3 alertas falsos injetados no SQLite do Edge.")