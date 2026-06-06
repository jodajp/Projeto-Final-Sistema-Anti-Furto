import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

# ==========================================
# 1. CONFIGURAÇÕES & URLs
# ==========================================
FALLBACK_SQLITE_URL = "sqlite:///./local_api.db"
MASTER_URL = os.environ.get("DATABASE_URL_MASTER") or os.environ.get("DATABASE_URL") or "postgresql://admin:123@db_master:5432/antifurto_db"
REPLICA_URL = os.environ.get("DATABASE_URL_REPLICA") or "postgresql://admin:123@db_replica:5432/antifurto_db"

# ==========================================
# 2. MOTOR DE BASE DE DADOS (ENGINE)
# ==========================================
def _build_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)

def _resolve_engine(database_url: str, fallback_url: str):
    """Tenta conectar ao DB principal. Se falhar, avisa e usa o fallback local."""
    engine = _build_engine(database_url or fallback_url)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"⚠️ Alerta: Falha ao conectar a {database_url}. A usar fallback SQLite local. Erro: {e}")
        return _build_engine(fallback_url)

# Criação dos Motores Master e Replica
engine_master = _resolve_engine(MASTER_URL, FALLBACK_SQLITE_URL)
engine_replica = _resolve_engine(REPLICA_URL, FALLBACK_SQLITE_URL)

# ==========================================
# 3. SESSÕES E MODELOS
# ==========================================
SessionMaster = sessionmaker(autocommit=False, autoflush=False, bind=engine_master)
SessionReplica = sessionmaker(autocommit=False, autoflush=False, bind=engine_replica)

Base = declarative_base()

# ==========================================
# 4. INJEÇÃO DE DEPENDÊNCIAS (FastAPI)
# ==========================================
def get_db_master():
    db = SessionMaster()
    try:
        yield db
    finally:
        db.close()

def get_db_replica():
    db = SessionReplica()
    try:
        yield db
    finally:
        db.close()