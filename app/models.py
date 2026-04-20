from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)  # whatsapp | standalone
    status: Mapped[str] = mapped_column(String(32), default="draft")
    source_text_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    merged_text_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    cancel_requested: Mapped[int] = mapped_column(Integer, default=0)
    transcription_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    transcription_finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    transcription_elapsed_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_transcription_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    audios: Mapped[list["Audio"]] = relationship("Audio", back_populates="project", cascade="all, delete-orphan")


class Audio(Base):
    __tablename__ = "audios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    transcript_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    transcript_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    line_hint: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    project: Mapped[Project] = relationship("Project", back_populates="audios")
