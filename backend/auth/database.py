from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from auth.config import AuthSettings


def _sqlite_connect_args(db_url: str) -> dict:
    if db_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


engine = create_engine(
    AuthSettings.DATABASE_URL,
    connect_args=_sqlite_connect_args(AuthSettings.DATABASE_URL),
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
