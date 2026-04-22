import json
import multiprocessing as mp
import os
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal
from app.models import Audio, Project
from app.services.audit import append_audit_log
from app.services.text_matching import merge_standalone_transcripts, merge_text_with_transcripts_inline, normalize_name
from app.services.transcriber import transcribe_file

_project_locks: dict[int, threading.Lock] = {}
_global_lock = threading.Lock()
_running_projects: set[int] = set()
_running_lock = threading.Lock()
_processes_lock = threading.Lock()
_project_processes: dict[int, mp.Process] = {}
_scheduler_lock = threading.Lock()
_gpu_probe_lock = threading.Lock()
_gpu_probe_cache: bool | None = None


def _get_project_lock(project_id: int) -> threading.Lock:
    with _global_lock:
        if project_id not in _project_locks:
            _project_locks[project_id] = threading.Lock()
        return _project_locks[project_id]


def _mark_project_running(project_id: int, running: bool) -> None:
    with _running_lock:
        if running:
            _running_projects.add(project_id)
        else:
            _running_projects.discard(project_id)


def is_project_running(project_id: int) -> bool:
    with _running_lock:
        return project_id in _running_projects


def _monitor_process(project_id: int, proc: mp.Process) -> None:
    proc.join()
    with _processes_lock:
        current = _project_processes.get(project_id)
        if current is proc:
            _project_processes.pop(project_id, None)
    _mark_project_running(project_id, False)
    trigger_scheduler()


def _running_process_count_unlocked() -> int:
    alive = 0
    stale: list[int] = []
    for pid, proc in _project_processes.items():
        if proc.is_alive():
            alive += 1
        else:
            stale.append(pid)
    for pid in stale:
        _project_processes.pop(pid, None)
    return alive


def _launch_project_process_unlocked(project_id: int) -> None:
    proc = mp.Process(target=_process_project, args=(project_id,), daemon=True)
    _project_processes[project_id] = proc
    _mark_project_running(project_id, True)
    proc.start()
    watcher = threading.Thread(target=_monitor_process, args=(project_id, proc), daemon=True)
    watcher.start()


def trigger_scheduler() -> None:
    if not _scheduler_lock.acquire(blocking=False):
        return
    try:
        limit = max(1, int(settings.transcription_project_concurrency or 1))
        db = SessionLocal()
        try:
            with _processes_lock:
                running = _running_process_count_unlocked()
                slots = max(0, limit - running)
                if slots == 0:
                    return

                queued = (
                    db.query(Project.id)
                    .filter(Project.status == "queued")
                    .order_by(Project.id.asc())
                    .limit(slots)
                    .all()
                )
                for (project_id,) in queued:
                    if project_id in _project_processes:
                        continue
                    _launch_project_process_unlocked(int(project_id))
        finally:
            db.close()
    finally:
        _scheduler_lock.release()


def _safe_stem(value: str) -> str:
    raw = Path(value).stem.strip().replace(" ", "_")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    cleaned = "".join(c for c in raw if c in allowed)
    return cleaned[:64] or "audio"


def _save_audio_transcription(audio: Audio, transcript_text: str, segments: list[dict], project_dir: Path) -> str:
    transcripts_dir = project_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(audio.original_name)
    txt_path = transcripts_dir / f"{audio.id}_{stem}.txt"
    json_path = transcripts_dir / f"{audio.id}_{stem}.json"

    txt_path.write_text(transcript_text + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    return str(txt_path)


def _is_cancel_requested(db: Session, project_id: int) -> bool:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        return True
    return bool(project.cancel_requested)


def _load_project_config(project_dir: Path) -> dict:
    cfg = project_dir / "config.json"
    if not cfg.exists():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_selected_ids(value) -> set[int]:
    ids: set[int] = set()
    if not isinstance(value, list):
        return ids
    for item in value:
        if isinstance(item, int):
            ids.add(item)
        elif isinstance(item, str) and item.isdigit():
            ids.add(int(item))
    return ids


def _update_audio_progress(audio_id: int, pct: int, status: str | None = None) -> None:
    db = SessionLocal()
    try:
        audio = db.query(Audio).filter(Audio.id == audio_id).first()
        if not audio:
            return
        audio.progress_pct = max(0, min(100, int(pct)))
        if status:
            audio.status = status
        db.commit()
    finally:
        db.close()


def _auto_worker_count(
    total_items: int,
    preferred_device: str | None,
    preferred_model_size: str | None,
    durations_seconds: list[float],
) -> int:
    cpu = os.cpu_count() or 4
    model = (preferred_model_size or "").lower()
    device = (preferred_device or "").lower()

    if device == "cuda":
        # GPU can saturate quickly; keep conservative default.
        guess = 1 if model in {"medium", "large-v3"} else 2
    else:
        if model in {"tiny", "base"}:
            guess = max(1, min(12, max(1, cpu - 1)))
        else:
            guess = max(1, min(8, cpu // 2))
        if model in {"medium", "large-v3"}:
            guess = max(1, min(guess, 2))

    known_durations = [float(v) for v in durations_seconds if isinstance(v, (int, float)) and float(v) > 0]
    if known_durations:
        avg_duration = sum(known_durations) / max(1, len(known_durations))
        if avg_duration >= 360:
            guess = max(1, min(guess, 2))
        elif avg_duration >= 180:
            guess = max(1, min(guess, 3))
        elif avg_duration <= 40:
            guess = min(max(guess, min(cpu, 8)), 12)

    return max(1, min(total_items, guess))


def _detect_gpu_available() -> bool:
    global _gpu_probe_cache
    with _gpu_probe_lock:
        if _gpu_probe_cache is not None:
            return _gpu_probe_cache
        try:
            result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
            _gpu_probe_cache = bool(result.returncode == 0 and result.stdout.strip())
        except Exception:
            _gpu_probe_cache = False
        return _gpu_probe_cache


def _duration_profile(duration_seconds: float | None) -> str:
    dur = float(duration_seconds or 0.0)
    if dur <= 45:
        return "short"
    if dur <= 180:
        return "medium"
    return "long"


def _auto_runtime_for_audio(duration_seconds: float | None) -> dict:
    profile = _duration_profile(duration_seconds)
    cpu = os.cpu_count() or 4
    gpu = _detect_gpu_available()

    if gpu:
        if profile == "short":
            return {
                "profile": profile,
                "model_size": "small",
                "device": "cuda",
                "compute_type": "float16",
                "chunk_seconds": 20,
                "enable_silence_trim": False,
            }
        if profile == "medium":
            return {
                "profile": profile,
                "model_size": "medium",
                "device": "cuda",
                "compute_type": "float16",
                "chunk_seconds": 35,
                "enable_silence_trim": False,
            }
        return {
            "profile": profile,
            "model_size": "medium",
            "device": "cuda",
            "compute_type": "float16",
            "chunk_seconds": 60,
            "enable_silence_trim": True,
        }

    if profile == "short":
        model = "base"
        if cpu >= 10:
            model = "small"
        return {
            "profile": profile,
            "model_size": model,
            "device": "cpu",
            "compute_type": "int8",
            "chunk_seconds": 20,
            "enable_silence_trim": False,
        }
    if profile == "medium":
        return {
            "profile": profile,
            "model_size": "small",
            "device": "cpu",
            "compute_type": "int8",
            "chunk_seconds": 35,
            "enable_silence_trim": False,
        }
    model = "small"
    if cpu >= 12:
        model = "medium"
    return {
        "profile": profile,
        "model_size": model,
        "device": "cpu",
        "compute_type": "int8",
        "chunk_seconds": 60,
        "enable_silence_trim": True,
    }


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


def _process_single_audio(
    project_id: int,
    audio_id: int,
    preferred_language: str | None,
    model_size: str | None,
    device: str | None,
    compute_type: str | None,
    project_dir: Path,
    parallel_workers: int | None = None,
):
    db = SessionLocal()
    try:
        if _is_cancel_requested(db, project_id):
            _update_audio_progress(audio_id, 0, "stopped")
            return

        audio = db.query(Audio).filter(Audio.id == audio_id).first()
        if not audio:
            return

        audio.status = "processing"
        audio.progress_pct = max(audio.progress_pct or 0, 1)
        db.commit()
        _update_audio_progress(audio_id, audio.progress_pct or 1, "processing")

        last_reported_pct = int(audio.progress_pct or 1)
        last_reported_at = time.monotonic()
        auto_runtime = _auto_runtime_for_audio(audio.duration_seconds)
        resolved_model_size = model_size or auto_runtime["model_size"]
        resolved_device = device or auto_runtime["device"]
        resolved_compute_type = compute_type or auto_runtime["compute_type"]
        runtime_profile = str(auto_runtime.get("profile") or "medium")
        chunk_seconds = int(auto_runtime.get("chunk_seconds") or settings.transcription_chunk_seconds)
        enable_silence_trim = bool(auto_runtime.get("enable_silence_trim"))

        def on_progress(pct: int):
            nonlocal last_reported_pct, last_reported_at
            local_db = SessionLocal()
            try:
                if _is_cancel_requested(local_db, project_id):
                    raise RuntimeError("cancel_requested")
            finally:
                local_db.close()
            now = time.monotonic()
            delta_pct = int(pct) - last_reported_pct
            if int(pct) < 100 and delta_pct < 1 and (now - last_reported_at) < 0.35:
                return
            _update_audio_progress(audio_id, pct, "processing")
            last_reported_pct = int(pct)
            last_reported_at = now

        try:
            text, segments = transcribe_file(
                Path(audio.stored_path),
                language=preferred_language,
                model_size=resolved_model_size,
                device=resolved_device,
                compute_type=resolved_compute_type,
                total_duration_seconds=audio.duration_seconds,
                progress_callback=on_progress,
                external_parallelism=parallel_workers,
                runtime_profile=runtime_profile,
                chunk_seconds=chunk_seconds,
                enable_silence_trim=enable_silence_trim,
            )
        except RuntimeError as exc:
            if str(exc) == "cancel_requested":
                _update_audio_progress(audio_id, audio.progress_pct or 0, "stopped")
                return
            raise

        if _is_cancel_requested(db, project_id):
            _update_audio_progress(audio_id, audio.progress_pct or 0, "stopped")
            return

        db.refresh(audio)
        audio.transcript_text = text
        audio.transcript_path = _save_audio_transcription(audio, text, segments, project_dir)
        audio.status = "done"
        audio.progress_pct = 100
        db.commit()
    finally:
        db.close()


def _process_project(project_id: int):
    lock = _get_project_lock(project_id)
    if not lock.acquire(blocking=False):
        return
    _mark_project_running(project_id, True)

    db: Session = SessionLocal()
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return

        if project.cancel_requested:
            project.status = "stopped"
            finished_at = datetime.now()
            _finalize_project_timing(project, finished_at)
            db.commit()
            append_audit_log(f"project={project_id} status=stopped_before_start")
            return

        project.cancel_requested = 0
        project.status = "processing"
        db.commit()
        append_audit_log(f"project={project_id} status=processing")

        project_dir = settings.data_dir / "projects" / str(project.id)
        config = _load_project_config(project_dir)
        preferred_language = config.get("language")
        preferred_model_size = config.get("model_size")
        preferred_device = config.get("device")
        preferred_compute_type = config.get("compute_type")
        preferred_workers = config.get("workers")
        selected_id_set = _parse_selected_ids(config.get("selected_audio_ids"))
        transcribe_only_referenced = bool(config.get("transcribe_only_referenced", False))

        audios = db.query(Audio).filter(Audio.project_id == project.id).order_by(Audio.id.asc()).all()
        if transcribe_only_referenced and not selected_id_set:
            selected_id_set = {int(audio.id) for audio in audios if bool(audio.line_hint)}

        target_audios: list[Audio | None] = []
        for audio in audios:
            should_skip = False
            if transcribe_only_referenced and audio.id not in selected_id_set:
                should_skip = True
            elif selected_id_set and audio.id not in selected_id_set:
                should_skip = True

            if should_skip:
                if audio.status not in {"done", "skipped"}:
                    audio.status = "skipped"
                    audio.progress_pct = 0
                target_audios.append(None)  # marker ignored
            else:
                # Resume behavior:
                # - keep successfully transcribed audios as-is (do not reprocess)
                # - only queue audios that are pending/incomplete/stopped/canceled/error/skipped
                transcript_ready = bool((audio.transcript_text or "").strip())
                if transcript_ready:
                    if audio.status != "done":
                        audio.status = "done"
                    audio.progress_pct = 100
                    target_audios.append(None)
                    continue

                if audio.status != "pending":
                    audio.status = "pending"
                if (audio.progress_pct or 0) >= 100:
                    audio.progress_pct = 0
                target_audios.append(audio)
        db.commit()

        todo = [a for a in target_audios if a is not None]
        if todo:
            if isinstance(preferred_workers, int):
                workers = max(1, min(int(preferred_workers), len(todo)))
            else:
                workers = _auto_worker_count(
                    len(todo),
                    preferred_device=preferred_device,
                    preferred_model_size=preferred_model_size,
                    durations_seconds=[float(a.duration_seconds or 0.0) for a in todo],
                )
            append_audit_log(f"project={project.id} workers={workers}")
            executor = ThreadPoolExecutor(max_workers=workers)
            canceling = False
            cancel_started_at: float | None = None
            try:
                pending = {
                    executor.submit(
                        _process_single_audio,
                        project.id,
                        a.id,
                        preferred_language,
                        preferred_model_size,
                        preferred_device,
                        preferred_compute_type,
                        project_dir,
                        workers,
                    )
                    for a in todo
                }

                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    for fut in done:
                        if fut.cancelled():
                            continue
                        fut.result()

                    db.expire_all()
                    fresh = db.query(Project).filter(Project.id == project.id).first()
                    if fresh and fresh.cancel_requested and not canceling:
                        fresh.status = "stopping"
                        db.commit()
                        canceling = True
                        cancel_started_at = time.monotonic()
                        executor.shutdown(wait=False, cancel_futures=True)
                        append_audit_log(f"project={project.id} executor_cancel_requested")

                    if canceling:
                        pending = {f for f in pending if not f.cancelled()}
                        if not pending:
                            break
                        if cancel_started_at and (time.monotonic() - cancel_started_at) > 3.0:
                            append_audit_log(f"project={project.id} executor_cancel_grace_timeout")
                            break
            finally:
                if not canceling:
                    executor.shutdown(wait=True, cancel_futures=False)

        db.expire_all()
        project = db.query(Project).filter(Project.id == project.id).first()
        audios = db.query(Audio).filter(Audio.project_id == project.id).order_by(Audio.id.asc()).all()

        transcript_by_audio: dict[str, str] = {}
        for audio in audios:
            if audio.status == "done" and (audio.transcript_text or "").strip():
                transcript_by_audio[normalize_name(audio.original_name)] = audio.transcript_text or ""

        output_dir = project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        if project.source_text_path:
            source_text = Path(project.source_text_path).read_text(encoding="utf-8")
            merged_text = merge_text_with_transcripts_inline(source_text, transcript_by_audio)
            merged_path = output_dir / "texto_com_transcricoes.txt"
            merged_path.write_text(merged_text, encoding="utf-8")
            project.merged_text_path = str(merged_path)
        else:
            summary_path = output_dir / "transcricoes_compiladas.txt"
            summary_path.write_text(merge_standalone_transcripts(transcript_by_audio), encoding="utf-8")
            project.merged_text_path = str(summary_path)

        if project.cancel_requested:
            project.status = "stopped"
            append_audit_log(f"project={project.id} status=stopped")
        else:
            project.status = "done"
            append_audit_log(f"project={project.id} status=done")

        finished_at = datetime.now()
        _finalize_project_timing(project, finished_at)

        db.commit()
    except Exception:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = "error"
            finished_at = datetime.now()
            _finalize_project_timing(project, finished_at)
            db.commit()
            append_audit_log(f"project={project.id} status=error")
        raise
    finally:
        _mark_project_running(project_id, False)
        db.close()
        lock.release()


def start_project_processing(project_id: int):
    with _processes_lock:
        current = _project_processes.get(project_id)
        if current and current.is_alive():
            return
    trigger_scheduler()
