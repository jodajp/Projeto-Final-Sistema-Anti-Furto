import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def _build_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, connect_args=connect_args)


def _resolve_engine(database_url: str, fallback_url: str):
    candidate = database_url or fallback_url
    engine = _build_engine(candidate)

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return engine
    except Exception:
        fallback_engine = _build_engine(fallback_url)
        return fallback_engine


# Usa as variáveis corretas do teu docker-compose, mas recorre a SQLite local
# quando o PostgreSQL Docker não está disponível (ex.: arranque local da API).
FALLBACK_SQLITE_URL = "sqlite:///./local_api.db"
MASTER_URL = os.environ.get("DATABASE_URL_MASTER") or os.environ.get("DATABASE_URL") or "postgresql://admin:123@db_master:5432/antifurto_db"
REPLICA_URL = os.environ.get("DATABASE_URL_REPLICA") or "postgresql://admin:123@db_replica:5432/antifurto_db"

engine_master = _resolve_engine(MASTER_URL, FALLBACK_SQLITE_URL)
engine_replica = _resolve_engine(REPLICA_URL, FALLBACK_SQLITE_URL)

SessionMaster = sessionmaker(autocommit=False, autoflush=False, bind=engine_master)
SessionReplica = sessionmaker(autocommit=False, autoflush=False, bind=engine_replica)

Base = declarative_base()

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