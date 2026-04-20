from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Whisper Chat Transcriber"
    data_dir: Path = Path("data")
    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    transcription_workers: int = 2
    transcription_project_concurrency: int = 1
    transcription_chunk_seconds: int = 45
    transcription_enable_silence_trim: bool = True
    transcription_cache_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "projects").mkdir(parents=True, exist_ok=True)
