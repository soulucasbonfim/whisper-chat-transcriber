import io
import json
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode
import asyncio
import tempfile
import time

from sqlalchemy import func
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slugify import slugify
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, SessionLocal, engine, ensure_runtime_schema, get_db
from app.models import Audio, Project
from app.services.audit import append_audit_log, append_aux_context_log
from app.services.media import human_duration, human_file_size, probe_audio_duration_seconds
from app.services.text_matching import extract_referenced_audio_names, normalize_name
from app.services.transcriber import transcribe_file
from app.services.worker import is_project_running, start_project_processing, trigger_scheduler

Base.metadata.create_all(bind=engine)
ensure_runtime_schema()

app = FastAPI(title=settings.app_name)
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)
templates = Jinja2Templates(directory="app/templates")

settings.data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/files", StaticFiles(directory=str(settings.data_dir)), name="files")


@app.on_event("startup")
def _startup_scheduler():
    trigger_scheduler()

ALLOWED_AUDIO_EXT = {".opus", ".ogg", ".mp3", ".m4a", ".aac", ".wav", ".flac"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm"}
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif"}
TEXT_DECODE_FALLBACKS = ["utf-8", "utf-8-sig", "latin-1"]
LANGUAGE_OPTIONS = {"auto", "pt", "en", "es", "fr", "de", "it"}
MODEL_OPTIONS = {"tiny", "base", "small", "medium", "large-v3"}
DEVICE_OPTIONS = {"cpu", "cuda"}
COMPUTE_OPTIONS = {"int8", "float16", "float32"}
DEFAULT_PROJECT_CONFIG = {
    "language": None,
    "extract_video_audio": False,
    "include_images_media": False,
    "include_videos_media": False,
    "include_stickers_media": False,
    "include_documents_media": False,
    "selected_audio_ids": [],
    "workers": None,
    "model_size": None,
    "device": None,
    "compute_type": None,
}
ACTIVE_STATUSES = {"queued", "processing", "stopping"}
IMPORT_ACTIVE_STATUSES = {"importing"}
REFRESH_STATUSES = ACTIVE_STATUSES | IMPORT_ACTIVE_STATUSES
IMPORT_STATE_FILENAME = "import_state.json"
MEDIA_INDEX_FILENAME = "media_index.json"
CONVERSATION_CACHE_FILENAME = "conversation_messages_cache.json"
ATTACHMENT_TOKEN_RE = re.compile(r"<(?:anexado|attached):\s*([^>]+)>", re.IGNORECASE)
_conversation_cache_lock = threading.Lock()
_conversation_cache: dict[str, dict] = {}
_conversation_cache_warmed_keys: dict[int, str] = {}
_conversation_cache_warm_inflight: dict[int, str] = {}


def _import_state_path(project_dir: Path) -> Path:
    return project_dir / IMPORT_STATE_FILENAME


def _media_index_path(project_dir: Path) -> Path:
    return project_dir / MEDIA_INDEX_FILENAME


def _load_media_index(project_dir: Path) -> list[dict]:
    path = _media_index_path(project_dir)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []
    except Exception:
        return []


def _save_media_index(project_dir: Path, items: list[dict]) -> None:
    path = _media_index_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def _public_file_url(stored_path: str, absolute_base_url: str | None = None) -> str:
    relative = stored_path.replace("\\", "/").replace("data/", "", 1).lstrip("/")
    if absolute_base_url:
        return absolute_base_url.rstrip("/") + "/files/" + relative
    return "/files/" + relative


def _write_import_state(
    project_dir: Path,
    progress_pct: int,
    message: str,
    stage: str = "importing",
    done: bool = False,
    error: str | None = None,
) -> None:
    payload = {
        "stage": stage,
        "progress_pct": max(0, min(100, int(progress_pct))),
        "message": message,
        "done": bool(done),
        "error": error,
        "updated_at": datetime.utcnow().isoformat(),
    }
    state_path = _import_state_path(project_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_import_state(project_dir: Path) -> dict:
    state_path = _import_state_path(project_dir)
    if not state_path.exists():
        return {"stage": None, "progress_pct": 0, "message": None, "done": True, "error": None}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("invalid import state payload")
        return {
            "stage": payload.get("stage"),
            "progress_pct": int(payload.get("progress_pct", 0)),
            "message": payload.get("message"),
            "done": bool(payload.get("done", False)),
            "error": payload.get("error"),
        }
    except Exception:
        return {"stage": None, "progress_pct": 0, "message": None, "done": False, "error": "estado_invalido"}


def is_audio_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_AUDIO_EXT


def is_video_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXT


def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXT


def is_sticker_file(filename: str) -> bool:
    lowered = Path(filename).name.lower()
    return "sticker" in lowered


def store_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return destination


def _unique_name(base_dir: Path, filename: str) -> str:
    stem = Path(filename).stem
    ext = Path(filename).suffix
    candidate = filename
    i = 2
    while (base_dir / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate


def _safe_storage_name(filename: str, fallback_prefix: str = "audio") -> str:
    ext = Path(filename).suffix.lower()
    stem = slugify(Path(filename).stem) or fallback_prefix
    return f"{stem}{ext}"


def _human_stem(filename: str | None, fallback: str) -> str:
    stem = Path(filename or "").stem.strip()
    if not stem:
        return fallback
    sanitized = re.sub(r"\s+", " ", stem).strip()
    return sanitized or fallback


def _sanitize_download_name(name: str, fallback: str) -> str:
    safe = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip())
    safe = re.sub(r"\s+", " ", safe).strip().strip(".")
    return safe or fallback


def _project_name_exists(db: Session, name: str) -> bool:
    normalized = (name or "").strip().lower()
    if not normalized:
        return False
    return (
        db.query(Project)
        .filter(func.lower(Project.name) == normalized)
        .first()
        is not None
    )


def _resolve_project_name(provided_name: str | None, fallback_name: str) -> str:
    candidate = (provided_name or "").strip()
    if candidate:
        return candidate
    return _human_stem(fallback_name, "Projeto sem nome")


def _redirect_home_with_error(message: str) -> RedirectResponse:
    query = urlencode({"error": message})
    return RedirectResponse(url=f"/?{query}", status_code=303)


def _decode_text_bytes(raw: bytes) -> str:
    for encoding in TEXT_DECODE_FALLBACKS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _normalize_language(language: str | None) -> str | None:
    if not language:
        return None
    normalized = language.strip().lower()
    if normalized not in LANGUAGE_OPTIONS:
        raise HTTPException(status_code=400, detail="Idioma invalido para transcricao.")
    if normalized == "auto":
        return None
    return normalized


def _normalize_workers(value: str | int | None) -> int | None:
    if value in (None, "", "auto"):
        return None
    workers = int(value)
    if workers < 1 or workers > 32:
        raise HTTPException(status_code=400, detail="Quantidade de workers invalida.")
    return workers


def _normalize_model_size(value: str | None) -> str | None:
    if not value or value == "auto":
        return None
    normalized = value.strip().lower()
    if normalized not in MODEL_OPTIONS:
        raise HTTPException(status_code=400, detail="Modelo Whisper invalido.")
    return normalized


def _normalize_device(value: str | None) -> str | None:
    if not value or value == "auto":
        return None
    normalized = value.strip().lower()
    if normalized not in DEVICE_OPTIONS:
        raise HTTPException(status_code=400, detail="Dispositivo invalido.")
    return normalized


def _normalize_compute_type(value: str | None) -> str | None:
    if not value or value == "auto":
        return None
    normalized = value.strip().lower()
    if normalized not in COMPUTE_OPTIONS:
        raise HTTPException(status_code=400, detail="Compute type invalido.")
    return normalized


def _read_total_memory_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (pages * page_size) / (1024**3)
            except Exception:
                pass
    return 8.0


def _read_available_memory_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().available / (1024**3)
    except Exception:
        pass

    # Linux fallback.
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / (1024**2)
    except Exception:
        pass

    # Conservative fallback.
    return max(0.5, _read_total_memory_gb() * 0.2)


def _runtime_pressure(cpu: int, total_mem_gb: float, available_mem_gb: float) -> dict:
    try:
        load1 = float(os.getloadavg()[0])
    except Exception:
        load1 = float(cpu) * 0.6
    load_ratio = load1 / max(1.0, float(cpu))
    mem_ratio = 0.0
    if total_mem_gb > 0:
        mem_ratio = max(0.0, min(1.0, available_mem_gb / total_mem_gb))
    return {
        "load1": round(load1, 2),
        "load_ratio": round(load_ratio, 3),
        "mem_available_gb": round(available_mem_gb, 2),
        "mem_ratio": round(mem_ratio, 3),
    }


def _adapt_workers_for_pressure(workers: int, cpu: int, total_mem_gb: float, available_mem_gb: float) -> tuple[int, dict]:
    workers = max(1, min(32, int(workers)))
    pressure = _runtime_pressure(cpu, total_mem_gb, available_mem_gb)

    load_ratio = float(pressure["load_ratio"])
    mem_ratio = float(pressure["mem_ratio"])
    adjusted = workers

    if load_ratio >= 0.95:
        adjusted = max(1, workers - 2)
    elif load_ratio >= 0.8:
        adjusted = max(1, workers - 1)

    if mem_ratio <= 0.08:
        adjusted = 1
    elif mem_ratio <= 0.15:
        adjusted = min(adjusted, 2)
    elif mem_ratio <= 0.25:
        adjusted = min(adjusted, 3)

    adjusted = max(1, min(adjusted, cpu))
    return adjusted, pressure


def _detect_gpu_available() -> bool:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass
    return False


def _runtime_benchmark_path() -> Path:
    return settings.data_dir / "runtime_benchmark.json"


def _load_runtime_benchmark() -> dict | None:
    path = _runtime_benchmark_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _save_runtime_benchmark(payload: dict) -> None:
    path = _runtime_benchmark_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _suggest_runtime_profile() -> dict:
    cpu = os.cpu_count() or 4
    mem_gb = _read_total_memory_gb()
    mem_avail_gb = _read_available_memory_gb()
    gpu = _detect_gpu_available()
    system = platform.system().lower()

    if gpu:
        workers = max(1, min(6, cpu // 2))
        model_size = "medium"
        device = "cuda"
        compute = "float16"
    else:
        if mem_gb >= 24 and cpu >= 10:
            workers = max(2, min(6, cpu - 2))
            model_size = "medium"
        elif mem_gb >= 12 and cpu >= 6:
            workers = max(2, min(4, cpu - 1))
            model_size = "small"
        else:
            workers = max(1, min(2, cpu))
            model_size = "base"
        device = "cpu"
        compute = "int8"

    benchmark = _load_runtime_benchmark()
    if benchmark:
        model_size = benchmark.get("model_size") or model_size
        device = benchmark.get("device") or device
        compute = benchmark.get("compute_type") or compute
        workers = int(benchmark.get("workers") or workers)

    workers, pressure = _adapt_workers_for_pressure(workers, cpu=cpu, total_mem_gb=mem_gb, available_mem_gb=mem_avail_gb)

    rationale = (
        f"cpu={cpu}, mem_gb={mem_gb:.1f}, mem_avail_gb={mem_avail_gb:.1f}, "
        f"load1={pressure['load1']}, gpu={gpu}, os={system}"
    )
    if benchmark:
        rationale += f", benchmark={benchmark.get('source','cached')}"
    return {
        "workers": workers,
        "model_size": model_size,
        "device": device,
        "compute_type": compute,
        "rationale": rationale,
        "runtime_pressure": pressure,
    }


def _run_quick_runtime_benchmark(db: Session) -> dict:
    # Uses a short clip from the first available audio to choose a practical
    # runtime profile for the current machine.
    audio = db.query(Audio).order_by(Audio.id.asc()).first()
    if not audio:
        raise HTTPException(status_code=400, detail="Nenhum áudio disponível para benchmark.")
    source = _safe_project_file(audio.stored_path)
    if not source:
        raise HTTPException(status_code=400, detail="Áudio de benchmark não encontrado.")

    baseline = _suggest_runtime_profile()
    candidates: list[dict] = []
    if baseline["device"] == "cuda":
        candidates = [
            {"model_size": "small", "device": "cuda", "compute_type": "float16"},
            {"model_size": "medium", "device": "cuda", "compute_type": "float16"},
        ]
    else:
        candidates = [
            {"model_size": "base", "device": "cpu", "compute_type": "int8"},
            {"model_size": "small", "device": "cpu", "compute_type": "int8"},
        ]

    best = None
    with tempfile.TemporaryDirectory(prefix="awhisper_bench_") as tmp:
        clip = Path(tmp) / "bench.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-t",
            "8",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(clip),
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if result.returncode != 0 or not clip.exists():
            raise HTTPException(status_code=400, detail="Falha ao preparar benchmark.")

        for candidate in candidates:
            t0 = time.perf_counter()
            try:
                transcribe_file(
                    clip,
                    language=None,
                    model_size=candidate["model_size"],
                    device=candidate["device"],
                    compute_type=candidate["compute_type"],
                    total_duration_seconds=8.0,
                    progress_callback=None,
                )
                elapsed = time.perf_counter() - t0
                item = dict(candidate)
                item["elapsed_seconds"] = round(elapsed, 3)
                if best is None or elapsed < best["elapsed_seconds"]:
                    best = item
            except Exception:
                continue

    if not best:
        raise HTTPException(status_code=500, detail="Benchmark não conseguiu avaliar configurações.")

    cpu = os.cpu_count() or 4
    total_mem_gb = _read_total_memory_gb()
    avail_mem_gb = _read_available_memory_gb()
    if best["device"] == "cuda":
        workers = max(1, min(3, cpu // 3))
    else:
        workers = max(1, min(6, cpu // 2))
        if best["model_size"] in {"small", "medium", "large-v3"}:
            workers = max(1, min(workers, 3))
    workers, pressure = _adapt_workers_for_pressure(
        workers,
        cpu=cpu,
        total_mem_gb=total_mem_gb,
        available_mem_gb=avail_mem_gb,
    )

    payload = {
        "source": "benchmark",
        "workers": workers,
        "model_size": best["model_size"],
        "device": best["device"],
        "compute_type": best["compute_type"],
        "elapsed_seconds": best["elapsed_seconds"],
        "runtime_pressure": pressure,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_runtime_benchmark(payload)
    append_audit_log(f"runtime_benchmark workers={workers} model={best['model_size']} device={best['device']}")
    return payload


def _load_project_config(project_dir: Path) -> dict:
    config_path = project_dir / "config.json"
    if not config_path.exists():
        return DEFAULT_PROJECT_CONFIG.copy()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return DEFAULT_PROJECT_CONFIG.copy()
        merged = DEFAULT_PROJECT_CONFIG.copy()
        merged.update(payload)
        if not isinstance(merged.get("selected_audio_ids"), list):
            merged["selected_audio_ids"] = []
        if bool(merged.get("include_non_audio_media", False)):
            merged["include_images_media"] = True
            merged["include_videos_media"] = True
            merged["include_stickers_media"] = True
            merged["include_documents_media"] = True
        return merged
    except Exception:
        return DEFAULT_PROJECT_CONFIG.copy()


def _save_project_config(project_dir: Path, payload: dict) -> None:
    config_path = project_dir / "config.json"
    merged = DEFAULT_PROJECT_CONFIG.copy()
    merged.update(payload)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_project_language(project_dir: Path) -> str:
    payload = _load_project_config(project_dir)
    lang = payload.get("language")
    return lang if lang else "auto"


def _best_text_member(members: list[zipfile.ZipInfo]) -> zipfile.ZipInfo | None:
    txts = [m for m in members if Path(m.filename).suffix.lower() == ".txt"]
    if not txts:
        return None

    def score(member: zipfile.ZipInfo) -> tuple[int, int, int]:
        name = Path(member.filename).name.lower()
        keyword = 0
        if name == "_chat.txt":
            keyword = 4
        elif "chat" in name:
            keyword = 3
        elif "conversa" in name or "whatsapp" in name:
            keyword = 2
        elif "txt" in name:
            keyword = 1
        return (keyword, member.file_size, -len(member.filename))

    return max(txts, key=score)


def _extract_audio_track_from_video(video_path: Path, target_audio_path: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target_audio_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0 and target_audio_path.exists()


def _extract_whatsapp_zip(
    archive_path: Path,
    project_dir: Path,
    include_video_audio: bool = False,
    include_images_media: bool = False,
    include_videos_media: bool = False,
    include_stickers_media: bool = False,
    include_documents_media: bool = False,
    progress_callback: Callable[[int, str], None] | None = None,
    on_audio_extracted: Callable[[dict[str, str]], None] | None = None,
) -> tuple[Path, list[dict[str, str]], list[dict[str, str]]]:
    source_dir = project_dir / "source"
    audio_dir = project_dir / "audios"
    media_dir = project_dir / "media"
    source_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    extracted_audios: list[dict[str, str]] = []
    extracted_media: list[dict[str, str]] = []

    with zipfile.ZipFile(archive_path) as zf:
        if progress_callback:
            progress_callback(6, "Lendo conteúdo do ZIP")
        members = [
            m
            for m in zf.infolist()
            if not m.is_dir() and "__MACOSX" not in m.filename and not Path(m.filename).name.startswith(".")
        ]

        text_member = _best_text_member(members)
        if not text_member:
            raise HTTPException(status_code=400, detail="Nao encontrei arquivo .txt dentro do zip.")

        text_raw = zf.read(text_member)
        decoded_text = _decode_text_bytes(text_raw)
        text_filename = _unique_name(source_dir, Path(text_member.filename).name)
        text_path = source_dir / text_filename
        text_path.write_text(decoded_text, encoding="utf-8")
        if progress_callback:
            progress_callback(16, "Texto da conversa identificado")

        def wants_member(member_name: str) -> bool:
            if is_audio_file(member_name):
                return True
            if is_video_file(member_name):
                return include_video_audio or include_videos_media
            if is_sticker_file(member_name):
                return include_stickers_media
            if is_image_file(member_name):
                return include_images_media
            if Path(member_name).suffix.lower() == ".txt":
                return False
            return include_documents_media

        media_candidates = [m for m in members if wants_member(Path(m.filename).name)]
        media_total = max(1, len(media_candidates))
        media_done = 0

        for member in members:
            original_name = Path(member.filename).name
            if not original_name:
                continue
            lower_suffix = Path(original_name).suffix.lower()

            if is_audio_file(original_name):
                safe = _safe_storage_name(original_name)
                safe = _unique_name(audio_dir, safe)
                stored_path = audio_dir / safe
                with zf.open(member) as source, stored_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                extracted_audios.append(
                    {
                        "original_name": original_name,
                        "stored_path": str(stored_path),
                    }
                )
                if on_audio_extracted:
                    on_audio_extracted(extracted_audios[-1])
                media_done += 1
                if progress_callback:
                    pct = 16 + int(round((media_done / media_total) * 64))
                    progress_callback(pct, f"Extraindo mídias ({media_done}/{media_total})")
                continue

            if is_video_file(original_name) and (include_video_audio or include_videos_media):
                safe_video = _safe_storage_name(original_name, fallback_prefix="video")
                safe_video = _unique_name(media_dir, safe_video)
                video_path = media_dir / safe_video
                with zf.open(member) as source, video_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

                if include_videos_media:
                    guessed, _ = mimetypes.guess_type(original_name)
                    extracted_media.append(
                        {
                            "original_name": original_name,
                            "stored_path": str(video_path),
                            "mime_type": guessed or "application/octet-stream",
                        }
                    )

                if include_video_audio:
                    audio_name = _unique_name(audio_dir, f"{Path(safe_video).stem}_from_video.wav")
                    audio_path = audio_dir / audio_name
                    if _extract_audio_track_from_video(video_path, audio_path):
                        extracted_audios.append(
                            {
                                "original_name": original_name,
                                "stored_path": str(audio_path),
                            }
                        )
                        if on_audio_extracted:
                            on_audio_extracted(extracted_audios[-1])
                media_done += 1
                if progress_callback:
                    pct = 16 + int(round((media_done / media_total) * 64))
                    progress_callback(pct, f"Extraindo mídias ({media_done}/{media_total})")
                continue

            should_store_non_audio = False
            if is_sticker_file(original_name):
                should_store_non_audio = include_stickers_media
            elif is_image_file(original_name):
                should_store_non_audio = include_images_media
            elif lower_suffix != ".txt" and not is_audio_file(original_name) and not is_video_file(original_name):
                should_store_non_audio = include_documents_media

            if should_store_non_audio:
                safe_media = _safe_storage_name(original_name, fallback_prefix="media")
                safe_media = _unique_name(media_dir, safe_media)
                media_path = media_dir / safe_media
                with zf.open(member) as source, media_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                guessed, _ = mimetypes.guess_type(original_name)
                extracted_media.append(
                    {
                        "original_name": original_name,
                        "stored_path": str(media_path),
                        "mime_type": guessed or "application/octet-stream",
                    }
                )
                media_done += 1
                if progress_callback:
                    pct = 16 + int(round((media_done / media_total) * 64))
                    progress_callback(pct, f"Extraindo mídias ({media_done}/{media_total})")

    if not extracted_audios:
        raise HTTPException(status_code=400, detail="Nao encontrei arquivos de audio dentro do zip.")

    return text_path, extracted_audios, extracted_media


def _create_project_record(db: Session, name: str, kind: str, initial_status: str = "draft") -> Project:
    final_name = (name or "").strip()
    if not final_name:
        raise HTTPException(status_code=400, detail="Nome do projeto inválido.")
    if _project_name_exists(db, final_name):
        raise HTTPException(status_code=400, detail="Já existe um projeto com esse nome.")

    project = Project(
        name=final_name,
        kind=kind,
        status=initial_status,
        cancel_requested=0,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    append_audit_log(f"project={project.id} created kind={kind}")
    return project


def _create_audio_records(
    db: Session,
    project_id: int,
    audios: list[dict[str, str]],
    source_text: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    known = [normalize_name(item["original_name"]) for item in audios]
    referenced = extract_referenced_audio_names(source_text, known) if source_text else set()

    count = 0
    total = len(audios)
    for item in audios:
        original_name = item["original_name"]
        norm = normalize_name(original_name)
        line_hint = "referenciado_no_texto" if norm in referenced else None
        stored = Path(item["stored_path"])
        file_size = stored.stat().st_size if stored.exists() else None
        duration = probe_audio_duration_seconds(stored)

        db.add(
            Audio(
                project_id=project_id,
                original_name=original_name,
                stored_path=item["stored_path"],
                status="pending",
                progress_pct=0,
                file_size_bytes=file_size,
                duration_seconds=duration,
                line_hint=line_hint,
            )
        )
        count += 1
        if progress_callback:
            progress_callback(count, total)
    db.commit()
    return count


def _create_audio_record_single(
    db: Session,
    project_id: int,
    item: dict[str, str],
    line_hint: str | None = None,
) -> int:
    original_name = item["original_name"]
    stored = Path(item["stored_path"])
    file_size = stored.stat().st_size if stored.exists() else None
    duration = probe_audio_duration_seconds(stored)
    row = Audio(
        project_id=project_id,
        original_name=original_name,
        stored_path=item["stored_path"],
        status="pending",
        progress_pct=0,
        file_size_bytes=file_size,
        duration_seconds=duration,
        line_hint=line_hint,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row.id


def _recompute_audio_line_hints(db: Session, project_id: int, source_text: str) -> None:
    audios = db.query(Audio).filter(Audio.project_id == project_id).all()
    known = [normalize_name(a.original_name or "") for a in audios]
    referenced = extract_referenced_audio_names(source_text, known) if source_text else set()
    for audio in audios:
        norm = normalize_name(audio.original_name or "")
        audio.line_hint = "referenciado_no_texto" if norm in referenced else None
    db.commit()


def _project_progress_pct(audios: list[Audio]) -> int:
    if not audios:
        return 0
    return int(round(sum((a.progress_pct or 0) for a in audios) / len(audios)))


def _audio_effectively_done_values(status: str | None, transcript_path: str | None, transcript_text: str | None = None) -> bool:
    if (status or "").lower() == "done":
        return True
    if (transcript_text or "").strip():
        return True
    if transcript_path:
        return True
    return False


def _audio_is_effectively_done(audio: Audio) -> bool:
    return _audio_effectively_done_values(audio.status, audio.transcript_path, audio.transcript_text)


def _selected_audio_ids_from_config(config: dict, audios: list[Audio]) -> set[int]:
    selected_ids = {
        int(v)
        for v in config.get("selected_audio_ids", [])
        if (isinstance(v, int) or (isinstance(v, str) and v.isdigit()))
    }
    if not selected_ids:
        return {a.id for a in audios}
    existing = {a.id for a in audios}
    return {aid for aid in selected_ids if aid in existing}


def _selected_audio_ids_from_existing(config: dict, existing_ids: set[int]) -> set[int]:
    selected_ids = {
        int(v)
        for v in config.get("selected_audio_ids", [])
        if (isinstance(v, int) or (isinstance(v, str) and v.isdigit()))
    }
    if not selected_ids:
        return set(existing_ids)
    return {aid for aid in selected_ids if aid in existing_ids}


def _is_project_fully_done(project: Project, db: Session) -> bool:
    audios = db.query(Audio).filter(Audio.project_id == project.id).all()
    if not audios:
        return False
    config = _load_project_config(settings.data_dir / "projects" / str(project.id))
    selected_ids = _selected_audio_ids_from_config(config, audios)
    if not selected_ids:
        return False
    for audio in audios:
        if audio.id not in selected_ids:
            continue
        if not _audio_is_effectively_done(audio):
            return False
    return True


def _project_status_display(project_status: str | None, project_fully_done: bool) -> tuple[str, str]:
    raw = (project_status or "draft").lower()
    if raw == "done" and not project_fully_done:
        return "incompleto", "incompleto"
    return raw, raw


def _start_button_label(project_status_display: str, has_partial_progress: bool = False) -> str:
    if project_status_display == "incompleto" or has_partial_progress:
        return "Continuar transcrição"
    return "Iniciar transcrição"


def _has_partial_progress_from_rows(rows: list, selected_ids: set[int] | None = None) -> bool:
    selected_filter = set(selected_ids or [])
    for row in rows:
        row_id = int(getattr(row, "id", 0) or 0)
        if selected_filter and row_id not in selected_filter:
            continue
        status = getattr(row, "status", None)
        transcript_path = getattr(row, "transcript_path", None)
        progress_pct = int(getattr(row, "progress_pct", 0) or 0)
        if _audio_effectively_done_values(status, transcript_path, None):
            return True
        if progress_pct > 0:
            return True
    return False


def _audio_status_rank(status: str | None) -> int:
    current = (status or "").lower()
    if current == "processing":
        return 0
    if current in {"pending", "queued", "stopping"}:
        return 1
    if current in {"error", "canceled", "stopped"}:
        return 2
    if current == "skipped":
        return 3
    if current == "done":
        return 9
    return 5


def _sort_audios_for_table(audios: list[Audio]) -> list[Audio]:
    return sorted(audios, key=lambda a: (_audio_status_rank(a.status), a.id))


def _format_datetime_br(value: datetime | None) -> str:
    if not value:
        return "-"
    return value.strftime("%d/%m/%Y %H:%M:%S")


def _resolve_run_elapsed_seconds(project: Project) -> float | None:
    if project.status in ACTIVE_STATUSES and project.transcription_started_at:
        return max(0.0, float((datetime.now() - project.transcription_started_at).total_seconds()))
    if project.transcription_elapsed_seconds is not None:
        return max(0.0, float(project.transcription_elapsed_seconds))
    return None


def _resolve_total_elapsed_seconds(project: Project) -> float | None:
    total = max(0.0, float(project.total_transcription_seconds or 0.0))
    if project.status in ACTIVE_STATUSES and project.transcription_started_at:
        total += max(0.0, float((datetime.now() - project.transcription_started_at).total_seconds()))
    return total if total > 0 else None


def _finalize_project_timing(project: Project, finished_at: datetime) -> float | None:
    run_elapsed: float | None = None
    if project.transcription_started_at:
        run_elapsed = max(0.0, float((finished_at - project.transcription_started_at).total_seconds()))

    previous_finished_at = project.transcription_finished_at
    project.transcription_finished_at = finished_at
    project.transcription_elapsed_seconds = run_elapsed

    if previous_finished_at is None and run_elapsed is not None:
        project.total_transcription_seconds = max(0.0, float(project.total_transcription_seconds or 0.0)) + run_elapsed

    return run_elapsed


def _project_status_payload(project: Project, db: Session) -> dict:
    project = _recover_stale_active_project(project, db)
    audio_rows = (
        db.query(
            Audio.id,
            Audio.original_name,
            Audio.status,
            Audio.progress_pct,
            Audio.transcript_path,
            Audio.line_hint,
            Audio.file_size_bytes,
            Audio.duration_seconds,
            Audio.stored_path,
        )
        .filter(Audio.project_id == project.id)
        .all()
    )
    audios = sorted(audio_rows, key=lambda r: (_audio_status_rank(r.status), r.id))
    project_dir = settings.data_dir / "projects" / str(project.id)
    config = _load_project_config(project_dir)
    import_state = _read_import_state(project_dir)
    progress_pct = _project_progress_pct(audios)
    if project.status == "importing":
        progress_pct = int(import_state.get("progress_pct") or 0)
    done_count = sum(1 for a in audios if (a.status == "done" or bool(a.transcript_path)))
    skipped_count = sum(1 for a in audios if a.status == "skipped")
    existing_ids = {int(a.id) for a in audios}
    selected_ids = _selected_audio_ids_from_existing(config, existing_ids)
    project_fully_done = bool(selected_ids)
    if project_fully_done:
        by_id = {int(a.id): a for a in audios}
        for aid in selected_ids:
            row = by_id.get(aid)
            if row is None or not _audio_effectively_done_values(row.status, row.transcript_path, None):
                project_fully_done = False
                break
    can_start = project.status in {"draft", "canceled", "stopped", "error", "import_error"} or (
        project.status == "done" and not project_fully_done
    )
    project_status_display, project_status_display_class = _project_status_display(project.status, project_fully_done)
    partial_progress = _has_partial_progress_from_rows(audios, None)
    start_button_label = _start_button_label(project_status_display, partial_progress)
    run_elapsed_seconds = _resolve_run_elapsed_seconds(project)
    total_elapsed_seconds = _resolve_total_elapsed_seconds(project)
    if project.status in {"draft", "done"}:
        _schedule_conversation_cache_warm_for_project(project)
    return {
        "project_id": project.id,
        "project_status": project.status,
        "project_status_display": project_status_display,
        "project_status_display_class": project_status_display_class,
        "project_fully_done": project_fully_done,
        "can_start": can_start,
        "start_button_label": start_button_label,
        "project_progress_pct": progress_pct,
        "merged_text_path": project.merged_text_path,
        "import_state": import_state,
        "footer_done_count": done_count,
        "footer_skipped_count": skipped_count,
        "footer_total_files": len(audios),
        "footer_selected_files": len(selected_ids),
        "transcription_started_at": _format_datetime_br(project.transcription_started_at),
        "transcription_finished_at": _format_datetime_br(project.transcription_finished_at),
        "transcription_run_elapsed_human": human_duration(run_elapsed_seconds),
        "transcription_total_elapsed_human": human_duration(total_elapsed_seconds),
        "selected_audio_ids": sorted(selected_ids),
        "audios": [
            {
                "id": a.id,
                "name": a.original_name,
                "status": a.status,
                "progress_pct": a.progress_pct or 0,
                "transcript_path": a.transcript_path,
                "line_hint": a.line_hint,
                "file_size_human": human_file_size(a.file_size_bytes),
                "duration_human": human_duration(a.duration_seconds),
                "audio_url": _public_file_url(a.stored_path) if a.stored_path else None,
                "selected": int(a.id) in selected_ids,
            }
            for a in audios
        ],
    }


def _status_summary(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "audios"}


def _audio_state_key(audio: dict) -> str:
    return "|".join(
        [
            str(audio.get("status") or ""),
            str(int(audio.get("progress_pct") or 0)),
            "1" if audio.get("line_hint") else "0",
            str(audio.get("transcript_path") or ""),
            "1" if audio.get("selected") else "0",
        ]
    )


def _audio_patch_compact(audio: dict) -> dict:
    return {
        "id": audio.get("id"),
        "status": audio.get("status"),
        "progress_pct": audio.get("progress_pct"),
        "line_hint": audio.get("line_hint"),
        "transcript_path": audio.get("transcript_path"),
        "selected": audio.get("selected"),
    }


def _recover_stale_active_project(project: Project, db: Session) -> Project:
    if project.status not in ACTIVE_STATUSES:
        return project
    if is_project_running(project.id):
        return project

    active_rows = (
        db.query(Audio)
        .filter(Audio.project_id == project.id, Audio.status.in_(["processing", "stopping", "queued"]))
        .all()
    )
    for audio in active_rows:
        audio.status = "canceled"

    if project.status == "queued":
        project.status = "draft"
        project.cancel_requested = 0
        append_audit_log(f"project={project.id} stale_queued_recovered_to_draft")
    else:
        project.status = "canceled"
        project.cancel_requested = 1
        finished_at = datetime.now()
        _finalize_project_timing(project, finished_at)
        append_audit_log(f"project={project.id} stale_active_recovered_to_canceled")

    db.commit()
    db.refresh(project)
    return project


def _run_zip_import(
    project_id: int,
    archive_path: Path,
    include_video_audio: bool,
    include_images_media: bool,
    include_videos_media: bool,
    include_stickers_media: bool,
    include_documents_media: bool,
    transcribe_only_referenced: bool,
) -> None:
    db = SessionLocal()
    project_dir = settings.data_dir / "projects" / str(project_id)
    registered_count = 0

    def on_audio_extracted(item: dict[str, str]) -> None:
        nonlocal registered_count
        _create_audio_record_single(db, project_id, item, line_hint=None)
        registered_count += 1
        _write_import_state(project_dir, 82, f"Indexando arquivos ({registered_count})")

    try:
        _write_import_state(project_dir, 4, "Iniciando leitura do ZIP")

        text_path, extracted_audios, extracted_media = _extract_whatsapp_zip(
            archive_path,
            project_dir,
            include_video_audio=include_video_audio,
            include_images_media=include_images_media,
            include_videos_media=include_videos_media,
            include_stickers_media=include_stickers_media,
            include_documents_media=include_documents_media,
            progress_callback=lambda pct, msg: _write_import_state(project_dir, pct, msg),
            on_audio_extracted=on_audio_extracted,
        )
        source_text = text_path.read_text(encoding="utf-8")
        _write_import_state(project_dir, 82, "Cadastrando áudios do projeto")

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return
        project.source_text_path = str(text_path)
        db.commit()

        _recompute_audio_line_hints(db, project_id, source_text)
        _write_import_state(project_dir, 96, "Indexação concluída")

        if transcribe_only_referenced:
            selected_ids = [
                a.id for a in db.query(Audio).filter(Audio.project_id == project_id, Audio.line_hint.is_not(None)).all()
            ]
            cfg = _load_project_config(project_dir)
            cfg["selected_audio_ids"] = selected_ids
            _save_project_config(project_dir, cfg)

        _save_media_index(project_dir, extracted_media)

        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = "draft"
            db.commit()
            _schedule_conversation_cache_warm_for_project(project)

        _write_import_state(project_dir, 100, "Importação concluída", stage="ready", done=True)
        append_audit_log(f"project={project_id} zip_imported audios={len(extracted_audios)}")
    except Exception as exc:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = "import_error"
            db.commit()
        _write_import_state(project_dir, 100, "Falha ao processar ZIP", stage="error", done=True, error=str(exc))
        append_audit_log(f"project={project_id} import_error={type(exc).__name__}")
    finally:
        db.close()


def _start_zip_import(
    project_id: int,
    archive_path: Path,
    include_video_audio: bool,
    include_images_media: bool,
    include_videos_media: bool,
    include_stickers_media: bool,
    include_documents_media: bool,
    transcribe_only_referenced: bool,
) -> None:
    thread = threading.Thread(
        target=_run_zip_import,
        args=(
            project_id,
            archive_path,
            include_video_audio,
            include_images_media,
            include_videos_media,
            include_stickers_media,
            include_documents_media,
            transcribe_only_referenced,
        ),
        daemon=True,
    )
    thread.start()


def _directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def _safe_project_file(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    try:
        resolved = path.resolve()
        data_root = settings.data_dir.resolve()
    except Exception:
        return None
    if data_root not in resolved.parents and resolved != data_root:
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


_WA_PATTERN_A = re.compile(
    r"^\[(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s*,?\s*(?P<time>\d{1,2}:\d{2})(?::\d{2})?\]\s*(?P<rest>.+)$"
)
_WA_PATTERN_B = re.compile(
    r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}),\s*(?P<time>\d{1,2}:\d{2})\s*-\s*(?P<rest>.+)$"
)
_WA_CONTROL_CHARS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def _normalize_wa_line(raw: str) -> str:
    # WhatsApp exports can include direction/control chars that break line parsing.
    return _WA_CONTROL_CHARS.sub("", raw).strip()


def _split_author(rest: str) -> tuple[str, str]:
    if ": " in rest:
        author, message = rest.split(": ", 1)
        return author.strip(), message.strip()
    return "Sistema", rest.strip()


def _parse_conversation_lines(text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = _normalize_wa_line(raw)
        if not line:
            continue

        match = _WA_PATTERN_A.match(line) or _WA_PATTERN_B.match(line)
        if match:
            author, message = _split_author(match.group("rest"))
            messages.append(
                {
                    "author": author or "Sistema",
                    "message": message or "",
                    "timestamp": f"{match.group('date')} {match.group('time')}",
                }
            )
            continue

        if messages:
            messages[-1]["message"] = (messages[-1]["message"] + "\n" + line).strip()
        else:
            messages.append({"author": "Sistema", "message": line, "timestamp": ""})
    return messages


def _build_media_lookup(project_dir: Path, audio_items: list[Audio] | None = None) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for item in _load_media_index(project_dir):
        original = item.get("original_name")
        stored = item.get("stored_path")
        if not original or not stored:
            continue
        key = normalize_name(str(original))
        if key and key not in lookup:
            mime_type = item.get("mime_type")
            if not mime_type:
                guessed, _ = mimetypes.guess_type(str(original))
                mime_type = guessed or "application/octet-stream"
            lookup[key] = {
                "original_name": str(original),
                "stored_path": str(stored),
                "mime_type": str(mime_type),
            }

    for audio in audio_items or []:
        if not audio.original_name or not audio.stored_path:
            continue
        guessed, _ = mimetypes.guess_type(audio.original_name)
        mime_type = guessed or "audio/*"
        keys = {
            normalize_name(str(audio.original_name)),
            normalize_name(Path(str(audio.stored_path)).name),
        }
        for key in keys:
            if not key or key in lookup:
                continue
            lookup[key] = {
                "original_name": str(audio.original_name),
                "stored_path": str(audio.stored_path),
                "mime_type": str(mime_type),
            }
    return lookup


def _decorate_message_with_attachment(
    message_text: str,
    media_lookup: dict[str, dict],
    absolute_base_url: str | None = None,
) -> dict:
    out = {
        "message": message_text or "",
        "attachment_name": None,
        "attachment_url": None,
        "attachment_kind": None,
        "attachment_stored_path": None,
    }
    match = ATTACHMENT_TOKEN_RE.search(message_text or "")
    if not match:
        return out

    attachment_name = (match.group(1) or "").strip()
    if not attachment_name:
        return out

    key = normalize_name(attachment_name)
    media = media_lookup.get(key)
    if not media:
        return out

    mime_type = (media.get("mime_type") or "application/octet-stream").lower()
    ext = Path(attachment_name).suffix.lower()
    kind = "document"
    if is_sticker_file(attachment_name):
        kind = "sticker"
    elif mime_type.startswith("image/") or ext in ALLOWED_IMAGE_EXT:
        kind = "image"
    elif mime_type.startswith("video/") or ext in ALLOWED_VIDEO_EXT:
        kind = "video"
    elif mime_type.startswith("audio/") or ext in ALLOWED_AUDIO_EXT:
        kind = "audio"
    elif ext == ".pdf":
        kind = "pdf"

    cleaned_message = ATTACHMENT_TOKEN_RE.sub("", message_text or "", count=1).strip()
    out["attachment_name"] = media.get("original_name") or attachment_name
    out["attachment_url"] = _public_file_url(media["stored_path"], absolute_base_url=absolute_base_url)
    out["attachment_kind"] = kind
    out["attachment_stored_path"] = media.get("stored_path")
    out["message"] = cleaned_message
    return out


def _decorate_chat_messages(
    messages: list[dict[str, str]],
    project_dir: Path,
    audio_items: list[Audio] | None = None,
    absolute_base_url: str | None = None,
) -> list[dict[str, str]]:
    side_map: dict[str, str] = {}
    order: list[str] = []
    decorated: list[dict[str, str]] = []
    media_lookup = _build_media_lookup(project_dir, audio_items=audio_items)
    for msg in messages:
        author_key = (msg.get("author") or "Sistema").strip().lower()
        if author_key not in side_map:
            if author_key in {"eu", "me", "you", "você", "voce"}:
                side_map[author_key] = "mine"
            else:
                idx = len(order)
                side_map[author_key] = "mine" if idx % 2 else "other"
                order.append(author_key)
        attachment = _decorate_message_with_attachment(
            msg.get("message") or "",
            media_lookup,
            absolute_base_url=absolute_base_url,
        )
        decorated.append(
            {
                "author": msg.get("author") or "Sistema",
                "message": attachment["message"],
                "timestamp": msg.get("timestamp") or "",
                "side": side_map[author_key],
                "attachment_name": attachment["attachment_name"],
                "attachment_url": attachment["attachment_url"],
                "attachment_kind": attachment["attachment_kind"],
                "attachment_stored_path": attachment["attachment_stored_path"],
            }
        )
    return decorated


def _build_conversation_bundle(
    project: Project, messages: list[dict], source_name: str, source_label: str, origin_label: str
) -> tuple[bytes, str]:
    base = _sanitize_download_name(project.name, f"projeto_{project.id}")
    bundle_root = slugify(base) or f"projeto_{project.id}"
    html_name = "abrir_conversa.html"

    rel_by_stored: dict[str, str] = {}
    name_counter: dict[str, int] = {}
    packaged_messages: list[dict] = []

    for msg in messages:
        cloned = dict(msg)
        stored = cloned.get("attachment_stored_path")
        original = cloned.get("attachment_name") or "anexo"
        if stored:
            key = str(stored)
            relative = rel_by_stored.get(key)
            if not relative:
                safe = _safe_storage_name(str(original), fallback_prefix="anexo")
                stem = Path(safe).stem
                ext = Path(safe).suffix
                count = name_counter.get(safe, 0) + 1
                name_counter[safe] = count
                if count > 1:
                    safe = f"{stem}_{count}{ext}"
                relative = f"attachments/{safe}"
                rel_by_stored[key] = relative
            cloned["attachment_url"] = relative
        packaged_messages.append(cloned)

    html = templates.env.get_template("conversation_export.html").render(
        project=project,
        messages=packaged_messages,
        source_name=source_name,
        source_label=source_label,
        origin_label=origin_label,
    )

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{bundle_root}/{html_name}", html)
        zf.writestr(
            f"{bundle_root}/LEIA-ME.txt",
            "Abra o arquivo 'abrir_conversa.html' para visualizar a conversa.\n",
        )

        for stored_path, relative in rel_by_stored.items():
            safe_file = _safe_project_file(stored_path)
            if not safe_file:
                continue
            zf.write(safe_file, arcname=f"{bundle_root}/{relative}")

    filename = f"{base}_conversa_com_anexos.zip"
    return output.getvalue(), filename


def _conversation_source_path(project: Project) -> Path:
    source_path = _safe_project_file(project.merged_text_path) or _safe_project_file(project.source_text_path)
    if not source_path:
        raise HTTPException(status_code=404, detail="Nenhum texto de conversa disponivel")
    return source_path


def _conversation_source_label(project: Project, source_path: Path) -> str:
    merged_path = _safe_project_file(project.merged_text_path)
    if merged_path and source_path == merged_path:
        return "texto original com transcrições"
    return "conversa original"


def _project_origin_label(project: Project) -> str:
    kind = (project.kind or "").strip().lower()
    if kind == "whatsapp_zip":
        return "ZIP (chat original)"
    if kind == "whatsapp":
        return "texto + áudios"
    if kind == "standalone":
        return "áudios avulsos"
    return kind or "desconhecido"


def _conversation_cache_key(project_id: int, source_path: Path) -> str:
    stat = source_path.stat()
    return f"{project_id}:{source_path}:{stat.st_mtime_ns}:{stat.st_size}"


def _conversation_disk_cache_path(project_id: int) -> Path:
    project_dir = settings.data_dir / "projects" / str(project_id)
    return project_dir / "output" / CONVERSATION_CACHE_FILENAME


def _load_disk_conversation_cache(project_id: int, key: str) -> list[dict] | None:
    path = _conversation_disk_cache_path(project_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("key") != key:
            return None
        messages = payload.get("messages")
        if isinstance(messages, list):
            return messages
    except Exception:
        return None
    return None


def _save_disk_conversation_cache(project_id: int, key: str, messages: list[dict]) -> None:
    path = _conversation_disk_cache_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "project_id": project_id,
        "updated_at": datetime.utcnow().isoformat(),
        "messages": messages,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _schedule_conversation_cache_warm_for_project(project: Project) -> None:
    source_path = _safe_project_file(project.merged_text_path) or _safe_project_file(project.source_text_path)
    if not source_path:
        return
    try:
        key = _conversation_cache_key(project.id, source_path)
    except Exception:
        return

    with _conversation_cache_lock:
        if _conversation_cache_warmed_keys.get(project.id) == key:
            return
        if _conversation_cache_warm_inflight.get(project.id) == key:
            return
        _conversation_cache_warm_inflight[project.id] = key

    def _runner() -> None:
        local_db = SessionLocal()
        try:
            fresh = local_db.query(Project).filter(Project.id == project.id).first()
            if fresh:
                _get_cached_conversation_messages(fresh, local_db)
        except Exception:
            pass
        finally:
            local_db.close()
            with _conversation_cache_lock:
                if _conversation_cache_warm_inflight.get(project.id) == key:
                    _conversation_cache_warm_inflight.pop(project.id, None)

    threading.Thread(target=_runner, name=f"conversation-warm-{project.id}", daemon=True).start()


def _get_cached_conversation_messages(project: Project, db: Session) -> tuple[Path, list[dict]]:
    source_path = _conversation_source_path(project)
    key = _conversation_cache_key(project.id, source_path)
    with _conversation_cache_lock:
        cached = _conversation_cache.get(key)
        if cached and isinstance(cached.get("messages"), list):
            _conversation_cache_warmed_keys[project.id] = key
            return source_path, cached["messages"]

    cached_disk = _load_disk_conversation_cache(project.id, key)
    if cached_disk is not None:
        with _conversation_cache_lock:
            stale_keys = [k for k in _conversation_cache.keys() if k.startswith(f"{project.id}:") and k != key]
            for k in stale_keys:
                _conversation_cache.pop(k, None)
            _conversation_cache[key] = {"messages": cached_disk}
            _conversation_cache_warmed_keys[project.id] = key
        return source_path, cached_disk

    project_dir = settings.data_dir / "projects" / str(project.id)
    raw = source_path.read_text(encoding="utf-8")
    parsed = _parse_conversation_lines(raw)
    project_audios = db.query(Audio).filter(Audio.project_id == project.id).all()
    messages = _decorate_chat_messages(parsed, project_dir=project_dir, audio_items=project_audios)

    with _conversation_cache_lock:
        # prune old cache entries for this project
        stale_keys = [k for k in _conversation_cache.keys() if k.startswith(f"{project.id}:") and k != key]
        for k in stale_keys:
            _conversation_cache.pop(k, None)
        _conversation_cache[key] = {"messages": messages}
        _conversation_cache_warmed_keys[project.id] = key

    _save_disk_conversation_cache(project.id, key, messages)

    return source_path, messages


@app.get("/")
def home(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).order_by(Project.created_at.desc()).all()
    project_total_size: dict[int, str] = {}
    for p in projects:
        size = _directory_size_bytes(settings.data_dir / "projects" / str(p.id))
        project_total_size[p.id] = human_file_size(size)
    error_message = request.query_params.get("error")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "projects": projects,
            "project_total_size": project_total_size,
            "error_message": error_message,
        },
    )


@app.get("/api/system/resource-suggestion")
def resource_suggestion():
    suggestion = _suggest_runtime_profile()
    return suggestion


@app.post("/api/system/resource-benchmark")
def run_resource_benchmark(db: Session = Depends(get_db)):
    return _run_quick_runtime_benchmark(db)


@app.post("/projects/whatsapp")
async def create_whatsapp_project(
    name: str = Form(""),
    transcription_language: str = Form("auto"),
    runtime_workers: str = Form("auto"),
    runtime_model_size: str = Form("auto"),
    runtime_device: str = Form("auto"),
    runtime_compute_type: str = Form("auto"),
    transcribe_only_referenced: bool = Form(False),
    conversation_file: UploadFile = File(...),
    audios: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    try:
        if not conversation_file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Envie um arquivo .txt para a conversa.")

        if not audios:
            raise HTTPException(status_code=400, detail="Envie pelo menos um audio.")

        resolved_name = _resolve_project_name(name, conversation_file.filename)
        project = _create_project_record(db, resolved_name, "whatsapp")

        project_dir = settings.data_dir / "projects" / str(project.id)
        source_dir = project_dir / "source"
        audio_dir = project_dir / "audios"
        language = _normalize_language(transcription_language)
        workers = _normalize_workers(runtime_workers)
        model_size = _normalize_model_size(runtime_model_size)
        device = _normalize_device(runtime_device)
        compute_type = _normalize_compute_type(runtime_compute_type)
        _save_project_config(
            project_dir,
            {
                "language": language,
                "extract_video_audio": False,
                "workers": workers,
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            },
        )

        safe_text_name = slugify(Path(conversation_file.filename).stem) or "conversa"
        text_path = source_dir / f"{safe_text_name}.txt"
        store_upload(conversation_file, text_path)
        project.source_text_path = str(text_path)

        source_text = text_path.read_text(encoding="utf-8")

        audio_payload: list[dict[str, str]] = []
        for audio in audios:
            if not audio.filename or not is_audio_file(audio.filename):
                continue

            safe_name = _safe_storage_name(audio.filename)
            safe_name = _unique_name(audio_dir, safe_name)
            stored_path = audio_dir / safe_name
            store_upload(audio, stored_path)
            audio_payload.append({"original_name": audio.filename, "stored_path": str(stored_path)})

        if not audio_payload:
            db.delete(project)
            db.commit()
            raise HTTPException(status_code=400, detail="Nenhum audio valido foi enviado.")

        _create_audio_records(db, project.id, audio_payload, source_text)
        if transcribe_only_referenced:
            selected_ids = [
                a.id for a in db.query(Audio).filter(Audio.project_id == project.id, Audio.line_hint.is_not(None)).all()
            ]
            cfg = _load_project_config(project_dir)
            cfg["selected_audio_ids"] = selected_ids
            _save_project_config(project_dir, cfg)
        db.commit()

        append_audit_log(f"project={project.id} audios={len(audio_payload)} uploaded")
        _schedule_conversation_cache_warm_for_project(project)
        return RedirectResponse(url=f"/projects/{project.id}", status_code=303)
    except HTTPException as exc:
        if 400 <= exc.status_code < 500:
            message = exc.detail if isinstance(exc.detail, str) else "Falha ao criar projeto."
            return _redirect_home_with_error(message)
        raise


@app.post("/projects/whatsapp-zip")
async def create_whatsapp_zip_project(
    name: str = Form(""),
    transcription_language: str = Form("auto"),
    runtime_workers: str = Form("auto"),
    runtime_model_size: str = Form("auto"),
    runtime_device: str = Form("auto"),
    runtime_compute_type: str = Form("auto"),
    include_video_audio: bool = Form(False),
    include_images_media: bool = Form(False),
    include_videos_media: bool = Form(False),
    include_stickers_media: bool = Form(False),
    include_documents_media: bool = Form(False),
    transcribe_only_referenced: bool = Form(False),
    export_zip: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        if not export_zip.filename.lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Envie um arquivo .zip de export do WhatsApp.")

        resolved_name = _resolve_project_name(name, export_zip.filename)
        project = _create_project_record(db, resolved_name, "whatsapp_zip", initial_status="importing")
        project_dir = settings.data_dir / "projects" / str(project.id)
        source_dir = project_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)
        language = _normalize_language(transcription_language)
        workers = _normalize_workers(runtime_workers)
        model_size = _normalize_model_size(runtime_model_size)
        device = _normalize_device(runtime_device)
        compute_type = _normalize_compute_type(runtime_compute_type)
        _save_project_config(
            project_dir,
            {
                "language": language,
                "extract_video_audio": include_video_audio,
                "include_images_media": include_images_media,
                "include_videos_media": include_videos_media,
                "include_stickers_media": include_stickers_media,
                "include_documents_media": include_documents_media,
                "workers": workers,
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            },
        )

        archive_name = _unique_name(source_dir, "export_whatsapp.zip")
        archive_path = source_dir / archive_name
        store_upload(export_zip, archive_path)
        _write_import_state(project_dir, 2, "Arquivo ZIP recebido")
        _start_zip_import(
            project.id,
            archive_path,
            include_video_audio,
            include_images_media,
            include_videos_media,
            include_stickers_media,
            include_documents_media,
            transcribe_only_referenced,
        )

        return RedirectResponse(url=f"/projects/{project.id}", status_code=303)
    except HTTPException as exc:
        if 400 <= exc.status_code < 500:
            message = exc.detail if isinstance(exc.detail, str) else "Falha ao criar projeto."
            return _redirect_home_with_error(message)
        raise


@app.post("/projects/standalone")
async def create_standalone_project(
    name: str = Form(""),
    transcription_language: str = Form("auto"),
    runtime_workers: str = Form("auto"),
    runtime_model_size: str = Form("auto"),
    runtime_device: str = Form("auto"),
    runtime_compute_type: str = Form("auto"),
    audios: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    try:
        if not audios:
            raise HTTPException(status_code=400, detail="Envie pelo menos um audio.")

        fallback_audio_name = ""
        for audio in audios:
            if audio.filename:
                fallback_audio_name = audio.filename
                break
        resolved_name = _resolve_project_name(name, fallback_audio_name or "audios")
        project = _create_project_record(db, resolved_name, "standalone")

        project_dir = settings.data_dir / "projects" / str(project.id)
        audio_dir = project_dir / "audios"
        language = _normalize_language(transcription_language)
        workers = _normalize_workers(runtime_workers)
        model_size = _normalize_model_size(runtime_model_size)
        device = _normalize_device(runtime_device)
        compute_type = _normalize_compute_type(runtime_compute_type)
        _save_project_config(
            project_dir,
            {
                "language": language,
                "extract_video_audio": False,
                "workers": workers,
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
            },
        )

        audio_payload: list[dict[str, str]] = []
        for audio in audios:
            if not audio.filename or not is_audio_file(audio.filename):
                continue

            safe_name = _safe_storage_name(audio.filename)
            safe_name = _unique_name(audio_dir, safe_name)
            stored_path = audio_dir / safe_name
            store_upload(audio, stored_path)
            audio_payload.append({"original_name": audio.filename, "stored_path": str(stored_path)})

        if not audio_payload:
            db.delete(project)
            db.commit()
            raise HTTPException(status_code=400, detail="Nenhum audio valido foi enviado.")

        _create_audio_records(db, project.id, audio_payload)
        append_audit_log(f"project={project.id} standalone_audios={len(audio_payload)}")
        return RedirectResponse(url=f"/projects/{project.id}", status_code=303)
    except HTTPException as exc:
        if 400 <= exc.status_code < 500:
            message = exc.detail if isinstance(exc.detail, str) else "Falha ao criar projeto."
            return _redirect_home_with_error(message)
        raise


@app.get("/projects/{project_id}")
def project_detail(project_id: int, request: Request, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")
    project = _recover_stale_active_project(project, db)

    audios = db.query(Audio).filter(Audio.project_id == project.id).order_by(Audio.id.asc()).all()
    audios = _sort_audios_for_table(audios)
    project_dir = settings.data_dir / "projects" / str(project.id)
    configured_language = _load_project_language(project_dir)
    config = _load_project_config(project_dir)
    import_state = _read_import_state(project_dir)
    project_progress_pct = _project_progress_pct(audios)
    if project.status == "importing":
        project_progress_pct = int(import_state.get("progress_pct") or 0)

    is_locked = project.status in REFRESH_STATUSES
    is_transcribing = project.status in ACTIVE_STATUSES
    project_fully_done = _is_project_fully_done(project, db)
    project_status_display, project_status_display_class = _project_status_display(project.status, project_fully_done)
    can_start = project.status in {"draft", "canceled", "stopped", "error", "import_error"} or (
        project.status == "done" and not project_fully_done
    )
    run_elapsed_seconds = _resolve_run_elapsed_seconds(project)
    total_elapsed_seconds = _resolve_total_elapsed_seconds(project)

    selected_audio_ids = _selected_audio_ids_from_config(config, audios)
    partial_progress = _has_partial_progress_from_rows(audios, None)
    start_button_label = _start_button_label(project_status_display, partial_progress)

    audio_rows = []
    total_audio_bytes = 0
    total_audio_duration = 0.0
    done_count = 0
    skipped_count = 0
    for a in audios:
        total_audio_bytes += int(a.file_size_bytes or 0)
        total_audio_duration += float(a.duration_seconds or 0.0)
        if _audio_is_effectively_done(a):
            done_count += 1
        if a.status == "skipped":
            skipped_count += 1
        audio_rows.append(
            {
                "id": a.id,
                "original_name": a.original_name,
                "audio_url": _public_file_url(a.stored_path) if a.stored_path else None,
                "status": a.status,
                "progress_pct": a.progress_pct or 0,
                "line_hint": a.line_hint,
                "transcript_path": a.transcript_path,
                "file_size_human": human_file_size(a.file_size_bytes),
                "duration_human": human_duration(a.duration_seconds),
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="project.html",
        context={
            "project": project,
            "audios": audio_rows,
            "configured_language": configured_language,
            "selected_audio_ids": selected_audio_ids,
            "extract_video_audio": bool(config.get("extract_video_audio", False)),
            "include_images_media": bool(config.get("include_images_media", False)),
            "include_videos_media": bool(config.get("include_videos_media", False)),
            "include_stickers_media": bool(config.get("include_stickers_media", False)),
            "include_documents_media": bool(config.get("include_documents_media", False)),
            "configured_workers": config.get("workers") or "auto",
            "configured_model_size": config.get("model_size") or "auto",
            "configured_device": config.get("device") or "auto",
            "configured_compute_type": config.get("compute_type") or "auto",
            "project_progress_pct": project_progress_pct,
            "project_status_display": project_status_display,
            "project_status_display_class": project_status_display_class,
            "is_locked": is_locked,
            "is_transcribing": is_transcribing,
            "can_start": can_start,
            "project_fully_done": project_fully_done,
            "start_button_label": start_button_label,
            "import_state": import_state,
            "footer_total_files": len(audios),
            "footer_selected_files": len(selected_audio_ids),
            "footer_done_count": done_count,
            "footer_skipped_count": skipped_count,
            "footer_total_size_human": human_file_size(total_audio_bytes),
            "footer_total_duration_human": human_duration(total_audio_duration),
            "footer_transcription_started_at": _format_datetime_br(project.transcription_started_at),
            "footer_transcription_finished_at": _format_datetime_br(project.transcription_finished_at),
            "footer_transcription_run_elapsed_human": human_duration(run_elapsed_seconds),
            "footer_transcription_total_elapsed_human": human_duration(total_elapsed_seconds),
        },
    )


@app.get("/projects/{project_id}/download-merged")
def download_merged_text(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")
    if not project.merged_text_path:
        raise HTTPException(status_code=404, detail="Texto final ainda nao foi gerado")

    merged_path = _safe_project_file(project.merged_text_path)
    if not merged_path:
        raise HTTPException(status_code=404, detail="Arquivo final nao encontrado")

    base = _sanitize_download_name(project.name, f"projeto_{project.id}")
    filename = f"{base}_texto_final.txt"
    return FileResponse(path=str(merged_path), media_type="text/plain; charset=utf-8", filename=filename)


@app.get("/projects/{project_id}/conversation")
def conversation_view(project_id: int, request: Request, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    source_path = _conversation_source_path(project)
    source_label = _conversation_source_label(project, source_path)
    origin_label = _project_origin_label(project)

    return templates.TemplateResponse(
        request=request,
        name="conversation.html",
        context={
            "project": project,
            "source_name": source_path.name,
            "source_label": source_label,
            "origin_label": origin_label,
        },
    )


@app.get("/api/projects/{project_id}/conversation/messages")
def conversation_messages_api(
    project_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(120, ge=1, le=500),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    _, messages = _get_cached_conversation_messages(project, db)
    total = len(messages)
    end = min(offset + limit, total)
    has_more = end < total
    return {
        "items": messages[offset:end],
        "offset": offset,
        "limit": limit,
        "next_offset": end if has_more else None,
        "has_more": has_more,
        "total": total,
    }


@app.get("/projects/{project_id}/conversation/download-html")
def conversation_download_html(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    source_path, messages = _get_cached_conversation_messages(project, db)
    source_label = _conversation_source_label(project, source_path)
    origin_label = _project_origin_label(project)
    exported_messages: list[dict] = []
    for msg in messages:
        cloned = dict(msg)
        attachment_name = (cloned.get("attachment_name") or "").strip()
        if attachment_name:
            marker = f"<anexado: {attachment_name}>"
            message_text = (cloned.get("message") or "").strip()
            cloned["message"] = f"{message_text}\n{marker}".strip()
        cloned["attachment_url"] = None
        cloned["attachment_kind"] = None
        cloned["attachment_stored_path"] = None
        exported_messages.append(cloned)

    html = templates.env.get_template("conversation_export.html").render(
        project=project,
        messages=exported_messages,
        source_name=source_path.name,
        source_label=source_label,
        origin_label=origin_label,
    )
    base = _sanitize_download_name(project.name, f"projeto_{project.id}")
    filename = f"{base}_conversa.html"
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/projects/{project_id}/conversation/download-bundle")
def conversation_download_bundle(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    source_path, messages = _get_cached_conversation_messages(project, db)
    source_label = _conversation_source_label(project, source_path)
    origin_label = _project_origin_label(project)

    bundle_bytes, bundle_name = _build_conversation_bundle(project, messages, source_path.name, source_label, origin_label)
    return Response(
        content=bundle_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{bundle_name}"'},
    )


@app.post("/projects/{project_id}/settings")
def update_project_settings(
    project_id: int,
    transcription_language: str = Form("auto"),
    runtime_workers: str = Form("auto"),
    runtime_model_size: str = Form("auto"),
    runtime_device: str = Form("auto"),
    runtime_compute_type: str = Form("auto"),
    include_video_audio: bool = Form(False),
    include_images_media: bool = Form(False),
    include_videos_media: bool = Form(False),
    include_stickers_media: bool = Form(False),
    include_documents_media: bool = Form(False),
    selected_audio_ids: list[int] = Form([]),
    db: Session = Depends(get_db),
):
    _update_project_settings_core(
        project_id=project_id,
        transcription_language=transcription_language,
        runtime_workers=runtime_workers,
        runtime_model_size=runtime_model_size,
        runtime_device=runtime_device,
        runtime_compute_type=runtime_compute_type,
        include_video_audio=include_video_audio,
        include_images_media=include_images_media,
        include_videos_media=include_videos_media,
        include_stickers_media=include_stickers_media,
        include_documents_media=include_documents_media,
        selected_audio_ids=selected_audio_ids,
        db=db,
    )
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


def _update_project_settings_core(
    project_id: int,
    transcription_language: str,
    runtime_workers: str,
    runtime_model_size: str,
    runtime_device: str,
    runtime_compute_type: str,
    include_video_audio: bool,
    include_images_media: bool,
    include_videos_media: bool,
    include_stickers_media: bool,
    include_documents_media: bool,
    selected_audio_ids: list[int],
    db: Session,
) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")
    if project.status in REFRESH_STATUSES:
        raise HTTPException(status_code=400, detail="Nao e possivel editar configuracao enquanto processa.")

    project_dir = settings.data_dir / "projects" / str(project.id)
    language = _normalize_language(transcription_language)
    workers = _normalize_workers(runtime_workers)
    model_size = _normalize_model_size(runtime_model_size)
    device = _normalize_device(runtime_device)
    compute_type = _normalize_compute_type(runtime_compute_type)
    existing_ids = {row[0] for row in db.query(Audio.id).filter(Audio.project_id == project.id).all()}
    sanitized = [int(v) for v in selected_audio_ids if int(v) in existing_ids]

    config = _load_project_config(project_dir)
    config["language"] = language
    config["workers"] = workers
    config["model_size"] = model_size
    config["device"] = device
    config["compute_type"] = compute_type
    config["extract_video_audio"] = include_video_audio
    config["include_images_media"] = include_images_media
    config["include_videos_media"] = include_videos_media
    config["include_stickers_media"] = include_stickers_media
    config["include_documents_media"] = include_documents_media
    config["selected_audio_ids"] = sanitized
    _save_project_config(project_dir, config)
    append_audit_log(f"project={project.id} settings_updated selected={len(sanitized)}")
    return project


@app.post("/api/projects/{project_id}/settings")
def update_project_settings_api(
    project_id: int,
    transcription_language: str = Form("auto"),
    runtime_workers: str = Form("auto"),
    runtime_model_size: str = Form("auto"),
    runtime_device: str = Form("auto"),
    runtime_compute_type: str = Form("auto"),
    include_video_audio: bool = Form(False),
    include_images_media: bool = Form(False),
    include_videos_media: bool = Form(False),
    include_stickers_media: bool = Form(False),
    include_documents_media: bool = Form(False),
    selected_audio_ids: list[int] = Form([]),
    db: Session = Depends(get_db),
):
    project = _update_project_settings_core(
        project_id=project_id,
        transcription_language=transcription_language,
        runtime_workers=runtime_workers,
        runtime_model_size=runtime_model_size,
        runtime_device=runtime_device,
        runtime_compute_type=runtime_compute_type,
        include_video_audio=include_video_audio,
        include_images_media=include_images_media,
        include_videos_media=include_videos_media,
        include_stickers_media=include_stickers_media,
        include_documents_media=include_documents_media,
        selected_audio_ids=selected_audio_ids,
        db=db,
    )
    return _project_status_payload(project, db)


@app.post("/projects/{project_id}/start")
def start_transcription(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    if project.status == "archived":
        raise HTTPException(status_code=400, detail="Projeto arquivado. Restaure para iniciar.")
    if project.status == "done" and _is_project_fully_done(project, db):
        raise HTTPException(status_code=400, detail="Projeto ja concluido. Nao e necessario iniciar novamente.")

    if project.status in REFRESH_STATUSES:
        return RedirectResponse(url=f"/projects/{project.id}", status_code=303)

    project.status = "queued"
    project.cancel_requested = 0
    project.transcription_started_at = datetime.now()
    project.transcription_finished_at = None
    project.transcription_elapsed_seconds = None
    db.commit()

    append_audit_log(f"project={project.id} start_requested")
    start_project_processing(project.id)
    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@app.post("/api/projects/{project_id}/start")
def start_transcription_api(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    if project.status == "archived":
        raise HTTPException(status_code=400, detail="Projeto arquivado. Restaure para iniciar.")
    if project.status == "done" and _is_project_fully_done(project, db):
        raise HTTPException(status_code=400, detail="Projeto ja concluido. Nao e necessario iniciar novamente.")

    if project.status not in REFRESH_STATUSES:
        project.status = "queued"
        project.cancel_requested = 0
        project.transcription_started_at = datetime.now()
        project.transcription_finished_at = None
        project.transcription_elapsed_seconds = None
        db.commit()
        append_audit_log(f"project={project.id} start_requested_api")
        start_project_processing(project.id)

    db.refresh(project)
    return _project_status_payload(project, db)


@app.post("/projects/{project_id}/stop")
def stop_transcription(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    if project.status in ACTIVE_STATUSES:
        project.cancel_requested = 1
        project.status = "stopping"
        active_audios = (
            db.query(Audio)
            .filter(Audio.project_id == project.id, Audio.status.in_(["queued", "pending", "processing"]))
            .all()
        )
        for audio in active_audios:
            if audio.status in {"queued", "pending"}:
                audio.status = "stopped"
                audio.progress_pct = 0
            elif audio.status == "processing":
                audio.status = "stopping"
        db.commit()
        append_audit_log(f"project={project.id} stop_requested")

    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@app.post("/api/projects/{project_id}/stop")
def stop_transcription_api(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    if project.status in ACTIVE_STATUSES:
        project.cancel_requested = 1
        project.status = "stopping"
        active_audios = (
            db.query(Audio)
            .filter(Audio.project_id == project.id, Audio.status.in_(["queued", "pending", "processing"]))
            .all()
        )
        for audio in active_audios:
            if audio.status in {"queued", "pending"}:
                audio.status = "stopped"
                audio.progress_pct = 0
            elif audio.status == "processing":
                audio.status = "stopping"
        db.commit()
        append_audit_log(f"project={project.id} stop_requested_api")

    db.refresh(project)
    return _project_status_payload(project, db)


@app.post("/projects/{project_id}/archive")
def archive_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")
    if project.status in REFRESH_STATUSES:
        raise HTTPException(status_code=400, detail="Nao e possivel arquivar enquanto processa.")

    project.status = "archived"
    db.commit()
    append_audit_log(f"project={project.id} archived")
    return RedirectResponse(url="/", status_code=303)


@app.post("/projects/{project_id}/restore")
def restore_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    if project.merged_text_path:
        project.status = "done"
    else:
        project.status = "draft"
    project.cancel_requested = 0
    db.commit()
    append_audit_log(f"project={project.id} restored")
    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@app.post("/projects/{project_id}/delete")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")
    if project.status in REFRESH_STATUSES:
        raise HTTPException(status_code=400, detail="Nao e possivel excluir enquanto processa.")

    project_dir = settings.data_dir / "projects" / str(project.id)
    db.delete(project)
    db.commit()
    shutil.rmtree(project_dir, ignore_errors=True)
    append_audit_log(f"project={project_id} deleted")
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/projects/{project_id}/status")
def project_status(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")
    return _project_status_payload(project, db)


@app.get("/api/projects/{project_id}/events")
async def project_status_events(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Projeto nao encontrado")

    async def event_stream():
        last_summary = ""
        last_audio_states: dict[int, str] = {}
        is_first = True
        idle_cycles = 0
        while True:
            local_db = SessionLocal()
            try:
                proj = local_db.query(Project).filter(Project.id == project_id).first()
                if not proj:
                    yield "event: close\ndata: {}\n\n"
                    break
                payload = _project_status_payload(proj, local_db)
            finally:
                local_db.close()

            summary = _status_summary(payload)
            encoded_summary = json.dumps(summary, ensure_ascii=False)
            current_audio_states = {int(a["id"]): _audio_state_key(a) for a in payload.get("audios", [])}

            if is_first:
                yield f"event: status_full\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                is_first = False
                last_summary = encoded_summary
                last_audio_states = current_audio_states
            else:
                changed_compact = []
                added_full = []
                if current_audio_states != last_audio_states:
                    by_id = {int(a["id"]): a for a in payload.get("audios", [])}
                    for audio_id, state_key in current_audio_states.items():
                        previous = last_audio_states.get(audio_id)
                        if previous == state_key:
                            continue
                        if previous is None:
                            added_full.append(by_id[audio_id])
                        else:
                            changed_compact.append(_audio_patch_compact(by_id[audio_id]))
                removed_ids = [audio_id for audio_id in last_audio_states.keys() if audio_id not in current_audio_states]

                has_delta = bool(changed_compact or added_full or removed_ids or encoded_summary != last_summary)
                if has_delta:
                    patch = dict(summary)
                    if changed_compact:
                        patch["audios_changed"] = changed_compact
                    if added_full:
                        patch["audios_added"] = added_full
                    if removed_ids:
                        patch["audios_removed"] = removed_ids
                    yield f"event: status_patch\ndata: {json.dumps(patch, ensure_ascii=False)}\n\n"
                    last_summary = encoded_summary
                    last_audio_states = current_audio_states
                else:
                    yield "event: ping\ndata: {}\n\n"

            if payload["project_status"] in REFRESH_STATUSES:
                idle_cycles = 0
            else:
                idle_cycles += 1
                if idle_cycles >= 3:
                    yield "event: close\ndata: {}\n\n"
                    break

            await asyncio.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


append_aux_context_log("Implementacao iniciada: controle de botoes, stop de transcricao, progresso e metadados")
