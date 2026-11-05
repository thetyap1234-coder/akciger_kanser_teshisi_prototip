"""
database.py - Esnek veritabanı katmanı (varsayılan: SQLite, istenirse PostgreSQL).

Kullanım:
  * Hiçbir şey ayarlamazsanız proje klasöründeki akciger.db (SQLite) kullanılır.
  * PostgreSQL için:
        set DATABASE_URL=postgresql+psycopg2://kullanici:sifre@localhost:5432/akciger_db
    (psycopg2 yalnızca bu durumda gerekir: pip install psycopg2-binary)
"""
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

_DEFAULT_DB = (Path(__file__).resolve().parent / "akciger.db").as_posix()
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DEFAULT_DB}")

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UserInDB(Base):
    """Kullanıcılar: hasta (patient) veya doktor (doctor)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    role = Column(String(16), default="patient", nullable=False)

    records = relationship("PredictionRecord", back_populates="user")


class PredictionRecord(Base):
    """Her analiz: girdiler, risk skoru, SHAP ve (varsa) görüntü füzyon sonucu."""

    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    risk_status = Column(String(32), nullable=False)
    risk_probability = Column(Float, nullable=False)
    inputs = Column(JSON, nullable=False)
    shap_contributions = Column(JSON, nullable=True)
    image_probability = Column(Float, nullable=True)
    final_risk_score = Column(Float, nullable=True)
    fusion_info = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    user = relationship("UserInDB", back_populates="records")


def _default_sql(col) -> str:
    """Mevcut satırlar için NOT NULL sütunlara sabit bir başlangıç değeri verir."""
    try:
        py_type = col.type.python_type
    except (AttributeError, NotImplementedError):
        py_type = str
    if isinstance(col.type, JSON) or py_type is dict:
        if engine.dialect.name == "postgresql":
            return " DEFAULT '{}'::json"
        return " DEFAULT '{}'"
    if py_type is int or py_type is float:
        return " DEFAULT 0"
    return " DEFAULT ''"


def _add_missing_columns() -> None:
    """Var olan tablolara modelde olup veritabanında olmayan sütunları ekler.

    create_all() mevcut tabloyu değiştirmez. Eski kayıtlardaki age/smoking/probability
    gibi sütunlar durur; yeni alanlar veri silinmeden eklenir.
    """
    insp = inspect(engine)
    preparer = engine.dialect.identifier_preparer
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        quoted_table = preparer.quote(table.name)
        for col in table.columns:
            if col.name in existing or col.primary_key:
                continue
            col_type = col.type.compile(dialect=engine.dialect)
            quoted_col = preparer.quote(col.name)
            default_sql = "" if col.nullable else _default_sql(col)
            null_sql = "" if col.nullable else " NOT NULL"
            ddl = f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_col} {col_type}{default_sql}{null_sql}"
            with engine.begin() as conn:
                conn.execute(text(ddl))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
