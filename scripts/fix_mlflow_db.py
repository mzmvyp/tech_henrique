# Fix MLflow DB: remove tabela temporaria que quebra a migracao
import sqlite3
import os
raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = os.path.join(raiz, "mlflow.db")
if not os.path.isfile(db):
    print("mlflow.db nao encontrado")
    exit(1)
conn = sqlite3.connect(db)
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [x[0] for x in cur.fetchall()]
print("Tabelas:", tables)
for t in tables:
    if "_alembic" in t or "tmp" in t.lower():
        cur.execute(f'DROP TABLE IF EXISTS "{t}"')
        print(f"Dropou {t}")
# Nome exato da tabela que quebra a migracao do MLflow
cur.execute("DROP TABLE IF EXISTS _alembic_tmp_latest_metrics")
conn.commit()
conn.close()
print("Pronto. Pode chamar /reload de novo.")
