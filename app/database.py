from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./data/app.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def ensure_runtime_schema():
    column_specs = {
        "projects": {
            "cancel_requested": "INTEGER DEFAULT 0",
            "transcription_started_at": "DATETIME",
            "transcription_finished_at": "DATETIME",
            "transcription_elapsed_seconds": "FLOAT",
            "total_transcription_seconds": "FLOAT DEFAULT 0",
        },
        "audios": {
            "progress_pct": "INTEGER DEFAULT 0",
            "file_size_bytes": "INTEGER",
            "duration_seconds": "FLOAT",
        },
    }

    with engine.begin() as conn:
        for table, specs in column_specs.items():
            existing = {row[1] for row in conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()}
            for col, ddl in specs.items():
                if col not in existing:
                    conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

        index_ddls = [
            "CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status)",
            "CREATE INDEX IF NOT EXISTS idx_projects_created_at ON projects(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_audios_project_status ON audios(project_id, status)",
            "CREATE INDEX IF NOT EXISTS idx_audios_project_id_id ON audios(project_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_audios_project_progress ON audios(project_id, progress_pct)",
        ]
        for ddl in index_ddls:
            conn.exec_driver_sql(ddl)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
