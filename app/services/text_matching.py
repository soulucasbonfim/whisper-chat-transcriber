import re
from pathlib import Path

AUDIO_RE = re.compile(r"[\w\-]+\.(?:opus|ogg|mp3|m4a|aac|wav|flac)", re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(name: str) -> str:
    return Path(name).name.strip().lower()


def _compact(value: str) -> str:
    return NON_ALNUM_RE.sub("", value.lower())


def find_referenced_audio(line: str, known_audio_names: list[str]) -> str | None:
    lowered = line.lower()
    compact_line = _compact(lowered)

    for name in known_audio_names:
        if name in lowered:
            return name

    for name in known_audio_names:
        stem = Path(name).stem.lower()
        compact_full = _compact(name)
        compact_stem = _compact(stem)
        if compact_full and compact_full in compact_line:
            return name
        if compact_stem and len(compact_stem) >= 8 and compact_stem in compact_line:
            return name

    for match in AUDIO_RE.findall(line):
        candidate = normalize_name(match)
        for known in known_audio_names:
            if candidate == known:
                return known

    return None


def merge_text_with_transcripts_inline(
    source_text: str,
    transcript_by_audio: dict[str, str],
) -> str:
    lines = source_text.splitlines()
    known_names = sorted(transcript_by_audio.keys(), key=len, reverse=True)
    used: set[str] = set()
    merged: list[str] = []

    for line in lines:
        ref = find_referenced_audio(line, known_names)
        if ref:
            transcript = " ".join((transcript_by_audio.get(ref) or "").split())
            if not transcript:
                transcript = "(sem fala detectada)"
            merged.append(f"{line} [TRANSCRITO: {ref} => {transcript}]")
            used.add(ref)
        else:
            merged.append(line)

    remaining = [name for name in transcript_by_audio.keys() if name not in used]
    if remaining:
        merged.append("")
        merged.append("[TRANSCRICOES SEM REFERENCIA DIRETA NO TEXTO]")
        merged.append("")
        for name in remaining:
            merged.append(f"{name} => {' '.join((transcript_by_audio[name] or '').split())}")
            merged.append("")

    return "\n".join(merged).strip() + "\n"


def extract_referenced_audio_names(source_text: str, known_audio_names: list[str]) -> set[str]:
    refs: set[str] = set()
    for line in source_text.splitlines():
        match = find_referenced_audio(line, known_audio_names)
        if match:
            refs.add(match)
    return refs


def merge_standalone_transcripts(transcript_by_audio: dict[str, str]) -> str:
    lines: list[str] = []
    for name, transcript in transcript_by_audio.items():
        lines.append(f"[{name}]")
        lines.append(" ".join((transcript or "").split()) or "(sem fala detectada)")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
