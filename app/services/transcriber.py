import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

from app.config import settings
from app.services.media import probe_audio_duration_seconds

_thread_local = threading.local()


def _cache_dir() -> Path:
    path = settings.data_dir / "transcription_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _cache_key(
    audio_path: Path,
    language: str | None,
    model_size: str | None,
    device: str | None,
    compute_type: str | None,
    runtime_profile: str | None,
    chunk_seconds: int | None,
    enable_silence_trim: bool | None,
) -> str:
    digest = _file_sha256(audio_path)
    suffix = "|".join(
        [
            language or "auto",
            model_size or settings.whisper_model_size,
            device or settings.whisper_device,
            compute_type or settings.whisper_compute_type,
            (runtime_profile or "auto").strip().lower() or "auto",
            str(int(chunk_seconds)) if chunk_seconds else "default_chunk",
            "trim_on" if enable_silence_trim is True else ("trim_off" if enable_silence_trim is False else "trim_default"),
        ]
    )
    return hashlib.sha256(f"{digest}|{suffix}".encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return _cache_dir() / f"{key}.json"


def _load_cached_result(key: str) -> tuple[str, list[dict[str, float | str]]] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = str(payload.get("text") or "").strip()
        segments = payload.get("segments") or []
        if not isinstance(segments, list):
            segments = []
        return text, segments
    except Exception:
        return None


def _save_cached_result(key: str, text: str, segments: list[dict[str, float | str]]) -> None:
    payload = {"text": text, "segments": segments, "saved_at": int(time.time())}
    _cache_path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _run_cmd(args: list[str]) -> bool:
    result = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return result.returncode == 0


def _trim_silence(source: Path, target: Path) -> bool:
    # Keeps speech and removes long silence before/after/interleaved pauses.
    return _run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-af",
            "silenceremove=stop_periods=-1:stop_duration=0.35:stop_threshold=-38dB",
            str(target),
        ]
    )


def _split_chunks(source: Path, out_dir: Path, seconds: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "chunk_%04d.wav"
    ok = _run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-f",
            "segment",
            "-segment_time",
            str(max(10, int(seconds))),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(pattern),
        ]
    )
    if not ok:
        return []
    return sorted(out_dir.glob("chunk_*.wav"))


def get_model(
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    external_parallelism: int | None = None,
) -> WhisperModel:
    ext_parallel = max(1, int(external_parallelism or 1))
    model_key = (
        model_size or settings.whisper_model_size,
        device or settings.whisper_device,
        compute_type or settings.whisper_compute_type,
        ext_parallel,
    )
    cache = getattr(_thread_local, "model_cache", None)
    if cache is None:
        cache = {}
        _thread_local.model_cache = cache

    model = cache.get(model_key)
    if model is None:
        model_device = model_key[1]
        whisper_kwargs: dict = {}
        if model_device == "cpu" and ext_parallel > 1:
            # Avoid CPU oversubscription when running multiple audio workers in parallel.
            whisper_kwargs["cpu_threads"] = 1
            whisper_kwargs["num_workers"] = 1
        elif model_device == "cpu":
            whisper_kwargs["cpu_threads"] = max(1, (os.cpu_count() or 4) - 1)

        model = WhisperModel(
            model_size_or_path=model_key[0],
            device=model_device,
            compute_type=model_key[2],
            **whisper_kwargs,
        )
        cache[model_key] = model
    return model


def transcribe_file(
    audio_path: Path,
    language: str | None = None,
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    total_duration_seconds: float | None = None,
    progress_callback: Callable[[int], None] | None = None,
    external_parallelism: int | None = None,
    runtime_profile: str | None = None,
    chunk_seconds: int | None = None,
    enable_silence_trim: bool | None = None,
) -> tuple[str, list[dict[str, float | str]]]:
    cache_key = ""
    if settings.transcription_cache_enabled:
        try:
            cache_key = _cache_key(
                audio_path,
                language,
                model_size,
                device,
                compute_type,
                runtime_profile,
                chunk_seconds,
                enable_silence_trim,
            )
            cached = _load_cached_result(cache_key)
            if cached:
                if progress_callback:
                    progress_callback(100)
                return cached
        except Exception:
            cache_key = ""

    model = get_model(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        external_parallelism=external_parallelism,
    )
    kwargs = {"vad_filter": True}
    if language:
        kwargs["language"] = language

    parsed_segments: list[dict[str, float | str]] = []
    full_text_chunks: list[str] = []
    last_reported = 0
    if progress_callback:
        progress_callback(1)

    with tempfile.TemporaryDirectory(prefix="awhisper_") as tmp:
        tmp_dir = Path(tmp)
        work_path = audio_path
        source_duration = total_duration_seconds or probe_audio_duration_seconds(audio_path)

        profile = (runtime_profile or "").strip().lower()
        if profile not in {"short", "medium", "long"}:
            if source_duration and float(source_duration) <= 45.0:
                profile = "short"
            elif source_duration and float(source_duration) <= 180.0:
                profile = "medium"
            else:
                profile = "long"

        short_audio = profile == "short"
        medium_audio = profile == "medium"

        if short_audio:
            # Fast path for short clips: reduce decoder search to improve latency.
            kwargs["vad_filter"] = False
            kwargs["beam_size"] = 1
            kwargs["best_of"] = 1
            kwargs["condition_on_previous_text"] = False
        elif medium_audio:
            kwargs["beam_size"] = 2
            kwargs["best_of"] = 2

        # Trimming silence improves quality in some files, but adds ffmpeg overhead.
        # Skip trimming for short/medium files to keep the pipeline faster.
        can_trim = source_duration is None or float(source_duration) >= 120.0
        trim_enabled = settings.transcription_enable_silence_trim if enable_silence_trim is None else bool(enable_silence_trim)
        if trim_enabled and can_trim:
            trimmed = tmp_dir / "trimmed.wav"
            if _trim_silence(audio_path, trimmed) and trimmed.exists() and trimmed.stat().st_size > 0:
                work_path = trimmed

        chunk_size = max(15, int(chunk_seconds or settings.transcription_chunk_seconds))
        # Splitting very short files usually hurts throughput; transcribe directly.
        effective_duration = source_duration or probe_audio_duration_seconds(work_path)
        should_split = effective_duration is None or float(effective_duration) > (chunk_size * 1.2)
        if should_split:
            chunks = _split_chunks(work_path, tmp_dir / "chunks", chunk_size)
            if not chunks:
                chunks = [work_path]
        else:
            chunks = [work_path]

        total_chunks = len(chunks)
        chunk_durations: list[float] = []
        for chunk in chunks:
            dur = probe_audio_duration_seconds(chunk)
            if dur is None or dur <= 0:
                dur = float(chunk_size)
            chunk_durations.append(float(dur))
        estimated_total = float(sum(chunk_durations)) if chunk_durations else 0.0
        progress_total_duration = (
            float(total_duration_seconds)
            if total_duration_seconds and total_duration_seconds > 0
            else (estimated_total if estimated_total > 0 else None)
        )

        time_offset = 0.0
        for idx, chunk_path in enumerate(chunks, start=1):
            segments, info = model.transcribe(str(chunk_path), **kwargs)
            chunk_duration = chunk_durations[idx - 1] if idx - 1 < len(chunk_durations) else float(chunk_size)
            chunk_has_segments = False
            for seg in segments:
                chunk_has_segments = True
                text = seg.text.strip()
                if text:
                    full_text_chunks.append(text)
                    parsed_segments.append(
                        {
                            "start": float(seg.start) + time_offset,
                            "end": float(seg.end) + time_offset,
                            "text": text,
                            "language": info.language,
                        }
                    )

                if progress_callback and progress_total_duration and progress_total_duration > 0:
                    absolute_end = max(0.0, float(seg.end) + time_offset)
                    pct = int(min(99, max(1, (absolute_end / progress_total_duration) * 100)))
                    if pct > last_reported:
                        progress_callback(pct)
                        last_reported = pct
                elif progress_callback:
                    chunk_pct = int(1 + (idx / max(1, total_chunks)) * 98)
                    if chunk_pct > last_reported:
                        progress_callback(chunk_pct)
                        last_reported = chunk_pct

            if progress_callback and progress_total_duration and progress_total_duration > 0 and not chunk_has_segments:
                chunk_end_pct = int(min(99, max(1, ((time_offset + chunk_duration) / progress_total_duration) * 100)))
                if chunk_end_pct > last_reported:
                    progress_callback(chunk_end_pct)
                    last_reported = chunk_end_pct

            # Approximate continuous timeline when chunk durations are unknown from model.
            try:
                chunk_dur = max((float(parsed_segments[-1]["end"]) - time_offset), 0.0) if parsed_segments else 0.0
            except Exception:
                chunk_dur = 0.0
            if chunk_dur <= 0.0:
                chunk_dur = chunk_duration
            time_offset += chunk_dur

    if progress_callback:
        progress_callback(100)

    full_text = " ".join(full_text_chunks).strip()
    if cache_key:
        try:
            _save_cached_result(cache_key, full_text, parsed_segments)
        except Exception:
            pass
    return full_text, parsed_segments
