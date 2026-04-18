#!/usr/bin/env python3
"""
mlx_whisper 전사 + whispermlx 화자 분리 병합 파이프라인

전사는 mlx_whisper의 high-quality 결과를 사용하고,
화자 분리는 whispermlx의 pyannote diarization을 사용해 병합한다.

업그레이드 포인트:
- word timestamp 기반 화자 배정
- 재실행을 빠르게 하는 artifact/cache 저장
- HF 토큰 환경변수 지원
- 긴 오디오 자동 chunking
- txt/json/srt/vtt 출력 지원

Usage:
    python3 transcribe.py meeting.m4a
    python3 transcribe.py meeting.m4a --diarize
    python3 transcribe.py meeting.m4a --diarize --formats txt,json,srt
    python3 transcribe.py meeting.m4a --diarize -o ./output
"""

import argparse
import atexit
from collections import Counter
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_TRANSCRIBE_MODEL = "mlx-community/whisper-large-v3-turbo"
MAX_QUALITY_TRANSCRIBE_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_QUALITY = "max"
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_OUTPUT_FORMATS = ("txt",)
SUPPORTED_OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")
HF_TOKEN_ENV_NAMES = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN")
UNKNOWN_SPEAKER = "UNKNOWN"
DEFAULT_CHUNK_MINUTES = 30.0
DEFAULT_AUTO_CHUNK_MINUTES = 60.0
DEFAULT_CHUNK_OVERLAP_SECONDS = 1.5
DEFAULT_SEGMENT_MERGE_GAP = 0.8
DEFAULT_SEGMENT_BREAK_GAP = 1.2
DEFAULT_LOCAL_CONFIG_NAME = ".transcribe.local.json"
DEFAULT_SHARED_GLOSSARY_NAME = "glossary.shared.json"
DEFAULT_ICLOUD_WHISPER_DIR = (
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Whisper"
)
DEFAULT_ICLOUD_INPUT_DIRNAME = "Whisper_input"
DEFAULT_ICLOUD_OUTPUT_DIRNAME = "Whisper_output"
QUALITY_CHOICES = ("fast", "standard", "max")
DEFAULT_PROGRESS_OUTPUTS = False
DEFAULT_RUN_LOCK_NAME = ".transcribe.lock"
DEFAULT_SHARED_MODEL_CACHE_DIRNAME = "_shared_models"
DEFAULT_REVIEW_TEMPERATURES = (0.0, 0.2, 0.4)
DEFAULT_REVIEW_BEST_OF = 4
SEGMENT_METADATA_KEYS = (
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "temperature",
    "seek",
    "id",
)
DEFAULT_REVIEW_PADDING_SECONDS = 1.0
DEFAULT_REVIEW_MAX_WINDOW_SECONDS = 12.0
DEFAULT_REVIEW_MAX_WINDOWS = 10
DEFAULT_REVIEW_MAX_TOTAL_SECONDS = 120.0
DEFAULT_REVIEW_CONTEXT_CHARS = 220
MAX_INITIAL_PROMPT_CHARS = 320
MAX_GLOSSARY_PROMPT_TERMS = 24
AUTO_GLOSSARY_STOPWORDS = {
    "a",
    "an",
    "and",
    "audio",
    "chunk",
    "dinner",
    "for",
    "from",
    "meeting",
    "review",
    "speaker",
    "stt",
    "the",
    "unknown",
}
AUTO_GLOSSARY_HANGUL_SUFFIXES = (
    "그룹",
    "금융",
    "기술",
    "랩스",
    "뱅크",
    "바이오",
    "벤처스",
    "산업",
    "생명",
    "시스템",
    "에너지",
    "은행",
    "전자",
    "증권",
    "카드",
    "캐피탈",
    "컴퍼니",
    "테크",
    "테크놀로지",
    "홀딩스",
    "화재",
)
AUTO_GLOSSARY_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&+./-]{1,}")
AUTO_GLOSSARY_HANGUL_RE = re.compile(
    rf"[가-힣]{{2,}}(?:{'|'.join(re.escape(suffix) for suffix in AUTO_GLOSSARY_HANGUL_SUFFIXES)})"
)
SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".wav",
    ".webm",
}


@dataclass
class ChunkSpec:
    index: int
    logical_start: float
    logical_end: float
    extract_start: float
    extract_end: float
    output_audio_path: Path
    is_chunked: bool


@dataclass
class OutputLayout:
    text_dir: Path
    structured_dir: Path
    artifact_root: Path


@dataclass
class GlossaryEntry:
    term: str
    aliases: Tuple[str, ...] = ()


@dataclass
class ReviewWindow:
    start: float
    end: float
    score: float
    reasons: List[str] = field(default_factory=list)
    source_indexes: List[int] = field(default_factory=list)


def project_data_dir() -> Path:
    return (Path.cwd() / "Data").resolve()


def default_input_dir() -> Path:
    candidates = (
        DEFAULT_ICLOUD_WHISPER_DIR / DEFAULT_ICLOUD_INPUT_DIRNAME,
        project_data_dir() / "input",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def default_output_dir() -> Path:
    candidates = (
        DEFAULT_ICLOUD_WHISPER_DIR / DEFAULT_ICLOUD_OUTPUT_DIRNAME,
        project_data_dir() / "output",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def log(message: str) -> None:
    print(message, flush=True)


def format_elapsed(seconds: float) -> str:
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def find_executable(name: str, extra_paths: Optional[Sequence[str]] = None) -> str:
    """PATH와 알려진 설치 경로에서 실행 파일 찾기"""
    import shutil

    found = shutil.which(name)
    if found:
        return found

    for path in extra_paths or []:
        expanded = os.path.expanduser(path)
        if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
            return expanded

    raise FileNotFoundError(f"{name} 실행 파일을 찾을 수 없습니다.")


def find_mlx_whisper() -> str:
    try:
        return find_executable(
            "mlx_whisper",
            extra_paths=[
                "~/Library/Python/3.9/bin/mlx_whisper",
                "~/Library/Python/3.10/bin/mlx_whisper",
                "~/Library/Python/3.11/bin/mlx_whisper",
                "~/Library/Python/3.12/bin/mlx_whisper",
                "~/.local/bin/mlx_whisper",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "mlx_whisper를 찾을 수 없습니다. `pip install mlx-whisper` 후 다시 시도하세요."
        ) from exc


def find_whispermlx() -> str:
    try:
        return find_executable(
            "whispermlx",
            extra_paths=[
                "~/.local/bin/whispermlx",
                "~/Library/Python/3.10/bin/whispermlx",
                "~/Library/Python/3.11/bin/whispermlx",
                "~/Library/Python/3.12/bin/whispermlx",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "whispermlx를 찾을 수 없습니다. `uv tool install whispermlx --python 3.12` 후 다시 시도하세요."
        ) from exc


def find_whispermlx_python() -> str:
    whispermlx_path = Path(find_whispermlx()).resolve()
    python_path = whispermlx_path.parent / "python"
    if python_path.is_file() and os.access(python_path, os.X_OK):
        return str(python_path)
    raise FileNotFoundError("whispermlx 전용 Python 실행 파일을 찾을 수 없습니다.")


def require_audio_file(audio_path: Path) -> None:
    if not audio_path.is_file():
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")


def is_supported_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


def collect_audio_files(audio_value: Optional[str], local_config: Dict) -> List[Path]:
    if audio_value:
        candidate = Path(audio_value).expanduser().resolve()
        if candidate.is_dir():
            audio_files = sorted(path for path in candidate.iterdir() if is_supported_audio_file(path))
            if not audio_files:
                raise FileNotFoundError(f"지원하는 오디오 파일이 없습니다: {candidate}")
            return audio_files
        require_audio_file(candidate)
        return [candidate]

    managed_input_dir = default_input_dir()
    if managed_input_dir.is_dir():
        audio_files = sorted(path for path in managed_input_dir.iterdir() if is_supported_audio_file(path))
        if audio_files:
            return audio_files

    config_audio = local_config.get("audio")
    if config_audio:
        candidate = Path(config_audio).expanduser().resolve()
        if candidate.is_dir():
            audio_files = sorted(path for path in candidate.iterdir() if is_supported_audio_file(path))
            if not audio_files:
                raise FileNotFoundError(f"지원하는 오디오 파일이 없습니다: {candidate}")
            return audio_files
        require_audio_file(candidate)
        return [candidate]

    raise RuntimeError(
        "오디오 파일 경로가 없습니다. "
        "CLI 인자/로컬 설정의 `audio`를 지정하거나 "
        f"`{managed_input_dir}`(호환 경로: `./Data/input`)에 오디오 파일을 넣어주세요."
    )


def resolve_output_dir(audio_files: Sequence[Path], output_dir_value: Optional[str], local_config: Dict) -> Path:
    managed_input_dir = default_input_dir()
    managed_output_dir = default_output_dir()
    files_from_default_input = (
        bool(audio_files)
        and all(path.parent == managed_input_dir for path in audio_files)
    )

    if output_dir_value:
        return Path(output_dir_value).expanduser().resolve()

    config_output_dir = local_config.get("output_dir")
    if files_from_default_input and config_output_dir in (None, ".", "./"):
        return managed_output_dir
    if config_output_dir:
        return Path(config_output_dir).expanduser().resolve()

    if files_from_default_input:
        return managed_output_dir

    return Path(".").resolve()


def build_output_layout(audio_files: Sequence[Path], output_dir: Path) -> OutputLayout:
    _ = audio_files
    managed_output_dir = default_output_dir()
    uses_managed_layout = output_dir == managed_output_dir

    if uses_managed_layout:
        data_root = project_data_dir()
        return OutputLayout(
            text_dir=managed_output_dir,
            structured_dir=data_root / "structured",
            artifact_root=data_root / "artifacts",
        )

    return OutputLayout(
        text_dir=output_dir,
        structured_dir=output_dir,
        artifact_root=output_dir,
    )


def require_ffmpeg() -> str:
    try:
        return find_executable(
            "ffmpeg",
            extra_paths=[
                "/opt/homebrew/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg를 찾을 수 없습니다. `brew install ffmpeg` 후 다시 시도하세요."
        ) from exc


def require_ffprobe() -> str:
    try:
        return find_executable(
            "ffprobe",
            extra_paths=[
                "/opt/homebrew/bin/ffprobe",
                "/usr/local/bin/ffprobe",
            ],
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffprobe를 찾을 수 없습니다. ffmpeg 설치를 확인하세요."
        ) from exc


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path: Path) -> Path:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                payload = {}
            existing_pid = int(payload.get("pid") or 0)
            if existing_pid and is_pid_running(existing_pid):
                raise RuntimeError(
                    "이미 실행 중인 전사 작업이 있습니다. "
                    "중복 실행은 매우 느려질 수 있으니 기존 작업을 먼저 종료해주세요."
                )
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "pid": current_pid,
                    "cwd": str(Path.cwd()),
                    "started_at": int(time.time()),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        break

    def cleanup() -> None:
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            payload = {}
        if int(payload.get("pid") or 0) == current_pid:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

    atexit.register(cleanup)
    return lock_path


def shared_model_cache_dir(layout: OutputLayout) -> Path:
    return layout.artifact_root / DEFAULT_SHARED_MODEL_CACHE_DIRNAME / "pyannote"


def run_preflight(diarize: bool) -> None:
    errors = []

    checks = [require_ffmpeg, require_ffprobe, find_mlx_whisper]
    if diarize:
        checks.append(find_whispermlx)

    for check in checks:
        try:
            check()
        except FileNotFoundError as exc:
            errors.append(str(exc))

    if errors:
        unique_errors = list(dict.fromkeys(errors))
        raise RuntimeError("필수 의존성이 없습니다:\n- " + "\n- ".join(unique_errors))


def run_cli(cmd: List[str], step_name: str) -> None:
    """외부 CLI 실행, 진행 로그는 실시간으로 그대로 보여준다."""
    try:
        completed = subprocess.run(
            cmd,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{step_name} 실패 (exit code: {exc.returncode})") from exc


def read_json(path: Path) -> Dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_formats(value: str) -> List[str]:
    raw = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw:
        return list(DEFAULT_OUTPUT_FORMATS)
    if "all" in raw:
        return list(SUPPORTED_OUTPUT_FORMATS)

    invalid = [fmt for fmt in raw if fmt not in SUPPORTED_OUTPUT_FORMATS]
    if invalid:
        raise ValueError(
            "지원하지 않는 출력 형식: "
            + ", ".join(invalid)
            + f" (지원: {', '.join(SUPPORTED_OUTPUT_FORMATS)})"
        )

    ordered = []
    for fmt in SUPPORTED_OUTPUT_FORMATS:
        if fmt in raw:
            ordered.append(fmt)
    return ordered


def resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value

    for env_name in HF_TOKEN_ENV_NAMES:
        token = os.getenv(env_name)
        if token:
            return token
    return None


def load_local_config(config_path: Path) -> Dict:
    if not config_path.is_file():
        return {}
    try:
        with config_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"로컬 설정 JSON을 읽지 못했습니다: {config_path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"로컬 설정 형식이 올바르지 않습니다: {config_path}")
    return data


def resolve_bool_option(current: Optional[bool], config: Dict, key: str, default: bool = False) -> bool:
    if current is not None:
        return current
    value = config.get(key, default)
    return bool(value)


def resolve_quality_option(current: Optional[str], config: Dict) -> str:
    if current is not None:
        return current

    value = config.get("quality", DEFAULT_QUALITY)
    if not isinstance(value, str):
        raise RuntimeError("로컬 설정의 `quality` 값은 문자열이어야 합니다.")

    normalized = value.strip().lower()
    if normalized not in QUALITY_CHOICES:
        raise RuntimeError(
            "로컬 설정의 `quality` 값은 " + ", ".join(QUALITY_CHOICES) + " 중 하나여야 합니다."
        )
    return normalized


def resolve_str_option(current: Optional[str], config: Dict, key: str) -> Optional[str]:
    if current:
        return current
    value = config.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeError(f"로컬 설정의 `{key}` 값은 문자열이어야 합니다.")
    return value


def parse_glossary_text(text: str) -> List[GlossaryEntry]:
    entries = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if not parts:
            continue
        entries.append(GlossaryEntry(term=parts[0], aliases=tuple(parts[1:])))
    return entries


def parse_glossary_value(value: object) -> List[GlossaryEntry]:
    entries: List[GlossaryEntry] = []
    if value is None:
        return entries
    if isinstance(value, str):
        return parse_glossary_text(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                entries.append(GlossaryEntry(term=item.strip()))
            elif isinstance(item, dict):
                term = str(item.get("term") or "").strip()
                if not term:
                    continue
                aliases_value = item.get("aliases") or []
                if isinstance(aliases_value, str):
                    aliases = (aliases_value.strip(),)
                else:
                    aliases = tuple(
                        str(alias).strip()
                        for alias in aliases_value
                        if str(alias).strip()
                    )
                entries.append(GlossaryEntry(term=term, aliases=aliases))
        return entries
    if isinstance(value, dict):
        for term, aliases_value in value.items():
            canonical = str(term).strip()
            if not canonical:
                continue
            if isinstance(aliases_value, str):
                aliases = (aliases_value.strip(),) if aliases_value.strip() else ()
            else:
                aliases = tuple(
                    str(alias).strip()
                    for alias in (aliases_value or [])
                    if str(alias).strip()
                )
            entries.append(GlossaryEntry(term=canonical, aliases=aliases))
        return entries
    raise RuntimeError("로컬 설정의 `glossary` 형식을 읽지 못했습니다.")


def load_glossary_file(path: Path, *, required: bool = True) -> List[GlossaryEntry]:
    if not path.is_file():
        if not required:
            return []
        raise FileNotFoundError(f"glossary 파일을 찾을 수 없습니다: {path}")
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return parse_glossary_value(payload)
    return parse_glossary_text(path.read_text(encoding="utf-8"))


def dedupe_glossary_entries(entries: Sequence[GlossaryEntry]) -> List[GlossaryEntry]:
    merged: Dict[str, List[str]] = {}
    for entry in entries:
        term = str(entry.term).strip()
        if not term:
            continue
        bucket = merged.setdefault(term, [])
        for alias in entry.aliases:
            cleaned = str(alias).strip()
            if cleaned and cleaned != term and cleaned not in bucket:
                bucket.append(cleaned)
    return [
        GlossaryEntry(term=term, aliases=tuple(aliases))
        for term, aliases in merged.items()
    ]


def default_shared_glossary_path() -> Path:
    return (Path.cwd() / DEFAULT_SHARED_GLOSSARY_NAME).resolve()


def resolve_shared_glossary_path(shared_glossary_value: Optional[str], config: Dict) -> Path:
    value = shared_glossary_value or config.get("shared_glossary_file")
    if value:
        return Path(str(value)).expanduser().resolve()
    return default_shared_glossary_path()


def serialize_glossary_entries(entries: Sequence[GlossaryEntry]) -> List[Dict[str, object]]:
    serialized = []
    for entry in entries:
        item: Dict[str, object] = {"term": entry.term}
        if entry.aliases:
            item["aliases"] = list(entry.aliases)
        serialized.append(item)
    return serialized


def write_glossary_file(path: Path, entries: Sequence[GlossaryEntry]) -> None:
    ordered = sorted(
        dedupe_glossary_entries(entries),
        key=lambda entry: (entry.term.lower(), entry.term),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        with path.open("w", encoding="utf-8") as handle:
            json.dump(serialize_glossary_entries(ordered), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return
    with path.open("w", encoding="utf-8") as handle:
        for entry in ordered:
            handle.write("|".join([entry.term, *entry.aliases]) + "\n")


def resolve_glossary_entries(
    args: argparse.Namespace,
    config: Dict,
    *,
    shared_glossary_path: Optional[Path] = None,
) -> List[GlossaryEntry]:
    entries: List[GlossaryEntry] = []
    config_glossary = config.get("glossary")
    if config_glossary is not None:
        entries.extend(parse_glossary_value(config_glossary))

    if shared_glossary_path:
        entries.extend(load_glossary_file(shared_glossary_path, required=False))

    glossary_file_value = args.glossary_file or config.get("glossary_file")
    if glossary_file_value:
        glossary_path = Path(str(glossary_file_value)).expanduser().resolve()
        entries.extend(load_glossary_file(glossary_path))

    if args.glossary_terms:
        entries.extend(parse_glossary_value(args.glossary_terms))

    return dedupe_glossary_entries(entries)


def build_glossary_prompt(entries: Sequence[GlossaryEntry]) -> Optional[str]:
    terms = [entry.term for entry in entries if entry.term][:MAX_GLOSSARY_PROMPT_TERMS]
    if not terms:
        return None
    return " ".join(terms)


def normalize_auto_glossary_term(value: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", str(value).strip())
    cleaned = re.sub(r"^[^0-9A-Za-z가-힣&+./-]+", "", cleaned)
    cleaned = re.sub(r"[^0-9A-Za-z가-힣&+./-]+$", "", cleaned)
    if not cleaned:
        return None
    if len(cleaned) < 2 or len(cleaned) > 40:
        return None

    core = re.sub(r"[^0-9A-Za-z가-힣]+", "", cleaned)
    if len(core) < 2 or core.isdigit():
        return None
    if cleaned.lower() in AUTO_GLOSSARY_STOPWORDS:
        return None
    return cleaned


def extract_filename_glossary_entries(audio_path: Path) -> List[GlossaryEntry]:
    stem = re.sub(r"[_-]?\d{6,8}$", "", audio_path.stem)
    candidates = []
    for raw_token in re.split(r"[^0-9A-Za-z가-힣&+./-]+", stem):
        token = normalize_auto_glossary_term(raw_token)
        if not token:
            continue
        if not re.search(r"[A-Za-z가-힣]", token):
            continue
        candidates.append(GlossaryEntry(term=token))
    return dedupe_glossary_entries(candidates)


def extract_transcript_glossary_entries(segments: Sequence[Dict]) -> List[GlossaryEntry]:
    counts: Counter[str] = Counter()

    for segment in segments:
        text = str(segment.get("text") or "")
        if not text:
            continue

        for raw_term in AUTO_GLOSSARY_LATIN_RE.findall(text):
            term = normalize_auto_glossary_term(raw_term)
            if not term:
                continue
            if not (any(char.isupper() for char in term) or term[0].isupper()):
                continue
            counts[term] += 1

        for raw_term in AUTO_GLOSSARY_HANGUL_RE.findall(text):
            term = normalize_auto_glossary_term(raw_term)
            if not term:
                continue
            counts[term] += 1

    extracted = []
    for term, count in counts.items():
        min_occurrences = 1 if any(char.isupper() for char in term) else 2
        if count >= min_occurrences:
            extracted.append(GlossaryEntry(term=term))

    return dedupe_glossary_entries(extracted)


def build_shared_glossary_seed_entries(
    audio_path: Path,
    segments: Sequence[Dict],
    args: argparse.Namespace,
) -> List[GlossaryEntry]:
    entries = list(getattr(args, "glossary_entries", []))
    entries.extend(extract_filename_glossary_entries(audio_path))
    entries.extend(extract_transcript_glossary_entries(segments))
    return dedupe_glossary_entries(entries)


def update_shared_glossary(
    audio_path: Path,
    segments: Sequence[Dict],
    args: argparse.Namespace,
) -> Tuple[Optional[Path], int]:
    shared_path = getattr(args, "shared_glossary_path", None)
    if not shared_path or not getattr(args, "shared_glossary_update", True):
        return None, 0

    existing_entries = dedupe_glossary_entries(load_glossary_file(shared_path, required=False))
    merged_entries = dedupe_glossary_entries(
        existing_entries + build_shared_glossary_seed_entries(audio_path, segments, args)
    )

    existing_signature = [(entry.term, entry.aliases) for entry in existing_entries]
    merged_signature = [(entry.term, entry.aliases) for entry in merged_entries]
    if merged_signature == existing_signature:
        return shared_path, 0

    write_glossary_file(shared_path, merged_entries)
    return shared_path, max(0, len(merged_entries) - len(existing_entries))


def compose_initial_prompt(
    args: argparse.Namespace,
    *,
    extra_prompt: Optional[str] = None,
) -> Optional[str]:
    parts = []
    if args.initial_prompt:
        parts.append(args.initial_prompt.strip())
    if extra_prompt:
        parts.append(extra_prompt.strip())
    glossary_prompt = build_glossary_prompt(getattr(args, "glossary_entries", []))
    if glossary_prompt:
        parts.append(glossary_prompt)
    prompt = " ".join(part for part in parts if part).strip()
    if not prompt:
        return None
    return prompt[:MAX_INITIAL_PROMPT_CHARS]


def mask_sensitive_command(command: str) -> str:
    masked = re.sub(r"(--hf-token\s+)(\S+)", r"\1***", command)
    masked = re.sub(r"(--hf_token\s+)(\S+)", r"\1***", masked)
    for env_name in HF_TOKEN_ENV_NAMES:
        masked = re.sub(rf"({env_name}=)(\S+)", r"\1***", masked)
    return masked


def find_output_json(output_dir: Path, preferred_stems: Optional[Sequence[str]] = None) -> Path:
    for stem in preferred_stems or []:
        candidate = output_dir / f"{stem}.json"
        if candidate.is_file():
            return candidate

    json_files = sorted(path for path in output_dir.iterdir() if path.suffix == ".json")
    if len(json_files) == 1:
        return json_files[0]

    raise FileNotFoundError(f"결과 JSON을 찾을 수 없습니다: {output_dir}")


def effective_transcribe_model(args: argparse.Namespace) -> str:
    if args.quality == "max" and args.transcribe_model == DEFAULT_TRANSCRIBE_MODEL:
        return MAX_QUALITY_TRANSCRIBE_MODEL
    return args.transcribe_model


def build_transcribe_decode_options(
    args: argparse.Namespace,
    *,
    word_timestamps: bool,
    extra_prompt: Optional[str] = None,
    review_pass: bool = False,
    language_override: Optional[str] = None,
) -> Dict[str, object]:
    is_max_quality = args.quality == "max"
    decode_options: Dict[str, object] = {
        "path_or_hf_repo": effective_transcribe_model(args),
        "verbose": False,
        # Keep the first pass deterministic; short review windows can afford a tiny fallback ladder.
        "temperature": (0.0, 0.2) if review_pass else 0.0,
        # Disabling previous-text conditioning reduces repetition loops on meeting audio.
        "condition_on_previous_text": False,
        "compression_ratio_threshold": 2.0 if is_max_quality else 2.4,
        "logprob_threshold": -0.7 if is_max_quality else -1.0,
        "no_speech_threshold": 0.6,
        "word_timestamps": word_timestamps,
        # Keep max quality practical on Apple Silicon by running inference in fp16.
        "fp16": True,
    }

    if word_timestamps:
        decode_options["hallucination_silence_threshold"] = 1.0

    runtime_language = language_override or args.language
    if runtime_language:
        decode_options["language"] = runtime_language
    initial_prompt = compose_initial_prompt(args, extra_prompt=extra_prompt)
    if initial_prompt:
        decode_options["initial_prompt"] = initial_prompt
    if review_pass:
        decode_options["temperature"] = DEFAULT_REVIEW_TEMPERATURES
        decode_options["best_of"] = DEFAULT_REVIEW_BEST_OF

    return decode_options


def build_transcribe_cache_stem(
    args: argparse.Namespace,
    *,
    word_timestamps: bool,
    extra_prompt: Optional[str] = None,
    review_pass: bool = False,
    clip_timestamps: Optional[str] = None,
    language_override: Optional[str] = None,
) -> str:
    decode_options = build_transcribe_decode_options(
        args,
        word_timestamps=word_timestamps,
        extra_prompt=extra_prompt,
        review_pass=review_pass,
        language_override=language_override,
    )
    if clip_timestamps:
        decode_options["clip_timestamps"] = clip_timestamps
    cache_payload = json.dumps(
        decode_options,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(cache_payload.encode("utf-8")).hexdigest()[:12]
    variant_root = "review" if review_pass else "mlx"
    variant_suffix = "words" if word_timestamps else "segments"
    return f"{variant_root}_{variant_suffix}_{digest}"


def describe_transcribe_runtime(args: argparse.Namespace, *, word_timestamps: bool) -> str:
    details = ["fp16", "greedy"]
    if args.quality == "max":
        details.append("no-prev-text")
    details.append("word-ts" if word_timestamps else "segment-ts")
    return ", ".join(details)


def warn_about_other_runs() -> List[Tuple[int, str]]:
    cmd = ["ps", "-Ao", "pid=,ppid=,command="]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []

    current_pid = os.getpid()
    parent_by_pid: Dict[int, int] = {}
    process_rows: List[Tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        parent_by_pid[pid] = ppid
        process_rows.append((pid, parts[2]))

    ancestor_pids = {current_pid}
    ancestor_pid = current_pid
    while True:
        parent_pid = parent_by_pid.get(ancestor_pid)
        if not parent_pid or parent_pid <= 1 or parent_pid in ancestor_pids:
            break
        ancestor_pids.add(parent_pid)
        ancestor_pid = parent_pid

    matches = []
    for pid, command in process_rows:
        if pid in ancestor_pids:
            continue
        if "caffeinate" in command and "transcribe.py" in command:
            continue
        if "transcribe.py" in command or "diarize_segments.py" in command:
            matches.append((pid, mask_sensitive_command(command)))

    if not matches:
        return []

    log("주의: 이미 실행 중인 전사/화자분리 프로세스가 있습니다. 병렬 실행은 매우 느려질 수 있습니다.")
    for pid, command in matches[:5]:
        log(f"- PID {pid}: {command}")
    return matches


def get_audio_duration(audio_path: Path) -> float:
    ffprobe = require_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("오디오 길이를 읽지 못했습니다.") from exc

    output = completed.stdout.strip()
    if not output:
        raise RuntimeError("오디오 길이를 읽지 못했습니다.")
    return float(output)


def plan_chunks(audio_path: Path, artifact_dir: Path, args: argparse.Namespace) -> Tuple[float, List[ChunkSpec]]:
    duration = get_audio_duration(audio_path)

    chunk_seconds = max(0.0, args.chunk_minutes * 60.0)
    auto_chunk_seconds = max(0.0, args.auto_chunk_minutes * 60.0)
    overlap = max(0.0, args.chunk_overlap_seconds)

    should_chunk = (
        not args.no_auto_chunk
        and chunk_seconds > 0
        and auto_chunk_seconds > 0
        and duration > auto_chunk_seconds
    )

    if not should_chunk:
        return duration, [
            ChunkSpec(
                index=0,
                logical_start=0.0,
                logical_end=duration,
                extract_start=0.0,
                extract_end=duration,
                output_audio_path=audio_path,
                is_chunked=False,
            )
        ]

    chunks_dir = artifact_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    logical_start = 0.0
    index = 0
    while logical_start < duration - 0.001:
        logical_end = min(duration, logical_start + chunk_seconds)
        extract_start = max(0.0, logical_start - (overlap if logical_start > 0 else 0.0))
        extract_end = min(duration, logical_end + (overlap if logical_end < duration else 0.0))
        chunks.append(
            ChunkSpec(
                index=index,
                logical_start=logical_start,
                logical_end=logical_end,
                extract_start=extract_start,
                extract_end=extract_end,
                output_audio_path=chunks_dir / f"chunk_{index:03d}.wav",
                is_chunked=True,
            )
        )
        logical_start = logical_end
        index += 1

    return duration, chunks


def materialize_chunk(audio_path: Path, chunk: ChunkSpec, force: bool) -> Path:
    if not chunk.is_chunked:
        return audio_path

    if chunk.output_audio_path.is_file() and not force:
        return chunk.output_audio_path

    ffmpeg = require_ffmpeg()
    chunk.output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, chunk.extract_end - chunk.extract_start)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{chunk.extract_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(chunk.output_audio_path),
    ]
    log(
        f"[chunk {chunk.index + 1}] 오디오 분할 중 "
        f"({format_clock(chunk.logical_start)} - {format_clock(chunk.logical_end)})..."
    )
    run_cli(cmd, f"chunk {chunk.index + 1} 오디오 분할")
    return chunk.output_audio_path


def prepare_work_audio(audio_path: Path, artifact_dir: Path, force: bool) -> Path:
    work_audio_path = artifact_dir / "_work_audio" / "source_16k_mono.wav"
    if work_audio_path.is_file() and not force:
        log(f"work audio cache 사용: {work_audio_path}")
        return work_audio_path

    ffmpeg = require_ffmpeg()
    work_audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(work_audio_path),
    ]
    log("공유 작업용 오디오 준비 중 (16kHz mono WAV)...")
    run_cli(cmd, "공유 작업용 오디오 준비")
    return work_audio_path


def load_or_run_mlx(
    audio_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    word_timestamps: bool,
    language_override: Optional[str] = None,
) -> Dict:
    output_stem = build_transcribe_cache_stem(
        args,
        word_timestamps=word_timestamps,
        language_override=language_override,
    )
    json_path = artifact_dir / "mlx" / f"{output_stem}.json"
    if json_path.is_file() and not args.force:
        log(f"mlx cache 사용: {json_path}")
        return read_json(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_whisper.transcribe import transcribe as mlx_transcribe
    except ImportError as exc:
        raise RuntimeError(
            "mlx_whisper Python 패키지를 import하지 못했습니다. `pip install mlx-whisper` 상태를 확인하세요."
        ) from exc

    decode_options = build_transcribe_decode_options(
        args,
        word_timestamps=word_timestamps,
        language_override=language_override,
    )
    result = mlx_transcribe(str(audio_path), **decode_options)
    write_json(json_path, result)
    return result


def load_or_run_diarization(
    audio_path: Path,
    artifact_dir: Path,
    hf_token: str,
    args: argparse.Namespace,
    *,
    model_cache_dir: Path,
) -> List[Dict]:
    output_dir = artifact_dir / "whispermlx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "diarization_intervals.json"
    if output_json.is_file() and not args.force:
        log(f"diarization cache 사용: {output_json}")
        payload = read_json(output_json)
        preferred_key = payload.get("preferred")
        if preferred_key and payload.get(preferred_key):
            return payload.get(preferred_key, [])
        return payload.get("exclusive_intervals") or payload.get("intervals", [])

    helper_script = Path(__file__).with_name("diarize_segments.py")
    whispermlx_python = find_whispermlx_python()
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        whispermlx_python,
        str(helper_script),
        str(audio_path),
        "--output-json",
        str(output_json),
        "--hf-token",
        hf_token,
        "--diarize-model",
        args.diarize_model,
        "--device",
        "mps",
        "--cache-dir",
        str(model_cache_dir),
    ]
    exact_speakers = getattr(args, "num_speakers", None)
    if exact_speakers is None and args.min_speakers is not None and args.min_speakers == args.max_speakers:
        exact_speakers = args.min_speakers
    if exact_speakers is not None:
        cmd.extend(["--num-speakers", str(exact_speakers)])
    elif args.min_speakers is not None:
        cmd.extend(["--min-speakers", str(args.min_speakers)])
    if exact_speakers is None and args.max_speakers is not None:
        cmd.extend(["--max-speakers", str(args.max_speakers)])

    run_cli(cmd, "pyannote 화자 분리")
    payload = read_json(output_json)
    preferred_key = payload.get("preferred")
    if preferred_key and payload.get(preferred_key):
        return payload.get(preferred_key, [])
    return payload.get("exclusive_intervals") or payload.get("intervals", [])


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def word_text(word: Dict) -> str:
    return str(word.get("word") or word.get("text") or "")


def rebuild_text_from_words(words: Sequence[Dict]) -> str:
    text = "".join(word_text(word) for word in words).strip()
    if text:
        return text
    return " ".join(word_text(word).strip() for word in words if word_text(word).strip()).strip()


def copy_segment_metadata(source: Dict, target: Dict) -> Dict:
    for key in SEGMENT_METADATA_KEYS:
        if key in source and source.get(key) is not None:
            target[key] = source[key]
    return target


def normalize_word(word: Dict) -> Dict:
    start = coerce_float(word.get("start"))
    end = coerce_float(word.get("end"), start)
    normalized = {
        "start": start,
        "end": max(start, end),
        "text": word_text(word),
    }
    if "speaker" in word and word.get("speaker") is not None:
        normalized["speaker"] = str(word["speaker"])
    if "score" in word and word.get("score") is not None:
        normalized["score"] = word["score"]
    return normalized


def normalize_segment(segment: Dict) -> Dict:
    start = coerce_float(segment.get("start"))
    end = coerce_float(segment.get("end"), start)
    normalized = copy_segment_metadata(segment, {
        "start": start,
        "end": max(start, end),
        "text": str(segment.get("text") or "").strip(),
    })
    if "speaker" in segment and segment.get("speaker") is not None:
        normalized["speaker"] = str(segment["speaker"])

    words = []
    for word in segment.get("words", []) or []:
        normalized_word = normalize_word(word)
        if normalized_word["text"]:
            words.append(normalized_word)

    if words:
        normalized["words"] = words
        normalized["start"] = words[0]["start"]
        normalized["end"] = words[-1]["end"]
        normalized["text"] = rebuild_text_from_words(words)

    return normalized


def normalize_result(payload: Dict) -> Dict:
    return {
        "language": payload.get("language"),
        "segments": [normalize_segment(segment) for segment in payload.get("segments", []) or []],
    }


def shift_segments(segments: Sequence[Dict], offset: float) -> List[Dict]:
    shifted = []
    for segment in segments:
        item = copy_segment_metadata(segment, {
            "start": segment["start"] + offset,
            "end": segment["end"] + offset,
            "text": segment.get("text", ""),
        })
        if "speaker" in segment:
            item["speaker"] = segment["speaker"]

        if segment.get("words"):
            item["words"] = []
            for word in segment["words"]:
                shifted_word = dict(word)
                shifted_word["start"] = word["start"] + offset
                shifted_word["end"] = word["end"] + offset
                item["words"].append(shifted_word)
            item["text"] = rebuild_text_from_words(item["words"])

        shifted.append(item)
    return shifted


def midpoint(start: float, end: float) -> float:
    if end <= start:
        return start
    return start + ((end - start) / 2.0)


def in_window_by_midpoint(start: float, end: float, window_start: float, window_end: float) -> bool:
    point = midpoint(start, end)
    if point == window_end and window_end > window_start:
        return False
    return window_start <= point < window_end


def trim_segments_to_window(segments: Sequence[Dict], window_start: float, window_end: float) -> List[Dict]:
    trimmed = []
    for segment in segments:
        words = []
        for word in segment.get("words", []) or []:
            if in_window_by_midpoint(word["start"], word["end"], window_start, window_end):
                words.append(dict(word))

        if words:
            item = copy_segment_metadata(segment, {
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": rebuild_text_from_words(words),
                "words": words,
            })
            if "speaker" in segment:
                item["speaker"] = segment["speaker"]
            trimmed.append(item)
            continue

        if not in_window_by_midpoint(segment["start"], segment["end"], window_start, window_end):
            continue

        item = copy_segment_metadata(segment, dict(segment))
        item["start"] = max(segment["start"], window_start)
        item["end"] = min(segment["end"], window_end)
        trimmed.append(item)

    return trimmed


def clone_segment(segment: Dict) -> Dict:
    item = dict(segment)
    if item.get("words"):
        item["words"] = [dict(word) for word in item["words"]]
    return item


def tokenize_quality_text(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [token for token in cleaned.split(" ") if token]


def longest_adjacent_token_run(tokens: Sequence[str]) -> int:
    if not tokens:
        return 0
    longest = 1
    current = 1
    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def longest_adjacent_phrase_run(tokens: Sequence[str], phrase_length: int) -> int:
    if phrase_length <= 0 or len(tokens) < phrase_length:
        return 0

    longest = 1
    max_start = len(tokens) - phrase_length
    for start in range(max_start + 1):
        phrase = tokens[start : start + phrase_length]
        if len(phrase) < phrase_length:
            continue
        run = 1
        cursor = start + phrase_length
        while cursor + phrase_length <= len(tokens) and tokens[cursor : cursor + phrase_length] == phrase:
            run += 1
            cursor += phrase_length
        longest = max(longest, run)
    return longest


def score_segment_for_review(segment: Dict) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []
    text = str(segment.get("text") or "").strip()
    duration = max(0.01, coerce_float(segment.get("end")) - coerce_float(segment.get("start")))

    avg_logprob = segment.get("avg_logprob")
    if avg_logprob is not None and avg_logprob < -0.85:
        score += min(2.0, (-0.85 - float(avg_logprob)) * 2.5)
        reasons.append(f"low-logprob:{float(avg_logprob):.2f}")

    compression_ratio = segment.get("compression_ratio")
    if compression_ratio is not None and compression_ratio > 1.55:
        score += min(2.0, (float(compression_ratio) - 1.55) * 2.5)
        reasons.append(f"high-compression:{float(compression_ratio):.2f}")

    tokens = tokenize_quality_text(text)
    max_repeat_run = longest_adjacent_token_run(tokens)
    if max_repeat_run >= 3:
        score += 1.0 + (0.35 * (max_repeat_run - 3))
        reasons.append(f"repeat-run:{max_repeat_run}")

    max_bigram_repeat_run = longest_adjacent_phrase_run(tokens, 2)
    if max_bigram_repeat_run >= 3:
        score += 1.4 + (0.55 * (max_bigram_repeat_run - 3))
        reasons.append(f"repeat-phrase-2:{max_bigram_repeat_run}")

    if tokens:
        short_token_ratio = sum(
            1 for token in tokens if len(re.sub(r"[^0-9A-Za-z가-힣]+", "", token)) <= 2
        ) / len(tokens)
        if len(tokens) >= 6 and short_token_ratio >= 0.7:
            score += 0.7
            reasons.append("short-token-heavy")
        token_cores = [
            re.sub(r"[^0-9A-Za-z가-힣]+", "", token).lower()
            for token in tokens
            if re.sub(r"[^0-9A-Za-z가-힣]+", "", token)
        ]
        if token_cores and len(tokens) >= 10:
            unique_ratio = len(set(token_cores)) / len(token_cores)
            if unique_ratio <= 0.45:
                score += 0.9
                reasons.append(f"low-unique:{unique_ratio:.2f}")

    if duration >= 8.0 and len(re.sub(r"\s+", "", text)) <= 10:
        score += 0.8
        reasons.append("low-density")

    if re.search(r"([가-힣A-Za-z])\1{4,}", text.replace(" ", "")):
        score += 0.8
        reasons.append("char-repeat")

    return score, reasons


def cap_review_window(window: ReviewWindow, duration: float) -> ReviewWindow:
    span = max(0.0, window.end - window.start)
    if span <= DEFAULT_REVIEW_MAX_WINDOW_SECONDS:
        return window
    center = window.start + (span / 2.0)
    half = DEFAULT_REVIEW_MAX_WINDOW_SECONDS / 2.0
    start = max(0.0, center - half)
    end = min(duration, start + DEFAULT_REVIEW_MAX_WINDOW_SECONDS)
    start = max(0.0, end - DEFAULT_REVIEW_MAX_WINDOW_SECONDS)
    return ReviewWindow(
        start=start,
        end=end,
        score=window.score,
        reasons=list(window.reasons),
        source_indexes=list(window.source_indexes),
    )


def collect_review_windows(segments: Sequence[Dict], duration: float) -> List[ReviewWindow]:
    candidates: List[ReviewWindow] = []
    for idx, segment in enumerate(segments):
        score, reasons = score_segment_for_review(segment)
        if score < 1.0:
            continue
        start = max(0.0, segment["start"] - DEFAULT_REVIEW_PADDING_SECONDS)
        end = min(duration, segment["end"] + DEFAULT_REVIEW_PADDING_SECONDS)
        candidates.append(
            cap_review_window(
                ReviewWindow(
                    start=start,
                    end=end,
                    score=score,
                    reasons=reasons,
                    source_indexes=[idx],
                ),
                duration,
            )
        )

    if segments and segments[0]["start"] >= 8.0:
        candidates.append(
            ReviewWindow(
                start=0.0,
                end=min(duration, min(segments[0]["start"] + 2.0, DEFAULT_REVIEW_MAX_WINDOW_SECONDS)),
                score=1.2,
                reasons=["leading-gap"],
                source_indexes=[],
            )
        )

    if not candidates:
        return []

    merged: List[ReviewWindow] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        if not merged or candidate.start > merged[-1].end + DEFAULT_REVIEW_PADDING_SECONDS:
            merged.append(candidate)
            continue

        previous = merged[-1]
        previous.end = max(previous.end, candidate.end)
        previous.score += candidate.score
        previous.reasons.extend(reason for reason in candidate.reasons if reason not in previous.reasons)
        previous.source_indexes.extend(
            idx for idx in candidate.source_indexes if idx not in previous.source_indexes
        )
        merged[-1] = cap_review_window(previous, duration)

    selected: List[ReviewWindow] = []
    total_review_seconds = 0.0
    for window in sorted(merged, key=lambda item: (-item.score, item.start)):
        span = max(0.0, window.end - window.start)
        if len(selected) >= DEFAULT_REVIEW_MAX_WINDOWS:
            break
        if selected and total_review_seconds + span > DEFAULT_REVIEW_MAX_TOTAL_SECONDS:
            continue
        selected.append(window)
        total_review_seconds += span

    return sorted(selected, key=lambda item: item.start)


def collect_segments_in_window(segments: Sequence[Dict], window_start: float, window_end: float) -> List[Dict]:
    return [
        clone_segment(segment)
        for segment in segments
        if in_window_by_midpoint(segment["start"], segment["end"], window_start, window_end)
    ]


def build_review_context_prompt(
    segments: Sequence[Dict],
    window_start: float,
    window_end: float,
) -> Optional[str]:
    previous_context = [
        str(segment.get("text") or "").strip()
        for segment in segments
        if segment["end"] <= window_start and str(segment.get("text") or "").strip()
    ][-2:]
    next_context = [
        str(segment.get("text") or "").strip()
        for segment in segments
        if segment["start"] >= window_end and str(segment.get("text") or "").strip()
    ][:1]
    context = " ".join(previous_context + next_context).strip()
    if not context:
        return None
    return context[:DEFAULT_REVIEW_CONTEXT_CHARS]


def load_or_run_review_window(
    audio_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    window: ReviewWindow,
    *,
    word_timestamps: bool,
    extra_prompt: Optional[str],
    language_override: Optional[str],
) -> Dict:
    clip_timestamps = f"{window.start:.2f},{window.end:.2f}"
    output_stem = build_transcribe_cache_stem(
        args,
        word_timestamps=word_timestamps,
        extra_prompt=extra_prompt,
        review_pass=True,
        clip_timestamps=clip_timestamps,
        language_override=language_override,
    )
    json_path = artifact_dir / "mlx" / f"{output_stem}.json"
    if json_path.is_file() and not args.force:
        log(f"review cache 사용: {json_path}")
        return read_json(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_whisper.transcribe import transcribe as mlx_transcribe
    except ImportError as exc:
        raise RuntimeError(
            "mlx_whisper Python 패키지를 import하지 못했습니다. `pip install mlx-whisper` 상태를 확인하세요."
        ) from exc

    decode_options = build_transcribe_decode_options(
        args,
        word_timestamps=word_timestamps,
        extra_prompt=extra_prompt,
        review_pass=True,
        language_override=language_override,
    )
    decode_options["clip_timestamps"] = clip_timestamps
    result = mlx_transcribe(str(audio_path), **decode_options)
    write_json(json_path, result)
    return result


def compute_segments_review_score(segments: Sequence[Dict]) -> float:
    if not segments:
        return 2.0
    return sum(score_segment_for_review(segment)[0] for segment in segments) / len(segments)


def replace_segments_in_window(
    segments: Sequence[Dict],
    window_start: float,
    window_end: float,
    replacement_segments: Sequence[Dict],
) -> List[Dict]:
    before = [
        clone_segment(segment)
        for segment in segments
        if segment["end"] <= window_start and not in_window_by_midpoint(segment["start"], segment["end"], window_start, window_end)
    ]
    after = [
        clone_segment(segment)
        for segment in segments
        if segment["start"] >= window_end and not in_window_by_midpoint(segment["start"], segment["end"], window_start, window_end)
    ]
    combined = before + [clone_segment(segment) for segment in replacement_segments] + after
    return sorted(combined, key=lambda item: (item["start"], item["end"]))


def refine_transcription_segments(
    audio_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    segments: Sequence[Dict],
    *,
    duration: float,
    word_timestamps: bool,
    language_override: Optional[str],
) -> Tuple[List[Dict], List[Dict]]:
    if not getattr(args, "review_pass", True):
        return [clone_segment(segment) for segment in segments], []

    review_windows = collect_review_windows(segments, duration)
    if not review_windows:
        return [clone_segment(segment) for segment in segments], []

    refined_segments = [clone_segment(segment) for segment in segments]
    review_records = []

    for review_idx, window in enumerate(review_windows, start=1):
        original_segments = collect_segments_in_window(refined_segments, window.start, window.end)
        context_prompt = build_review_context_prompt(refined_segments, window.start, window.end)
        reviewed_payload = normalize_result(
            load_or_run_review_window(
                audio_path,
                artifact_dir,
                args,
                window,
                word_timestamps=word_timestamps,
                extra_prompt=context_prompt,
                language_override=language_override,
            )
        )
        reviewed_segments = trim_segments_to_window(
            reviewed_payload["segments"],
            window.start,
            window.end,
        )

        original_score = compute_segments_review_score(original_segments)
        reviewed_score = compute_segments_review_score(reviewed_segments)
        original_text = " ".join(segment.get("text", "").strip() for segment in original_segments).strip()
        reviewed_text = " ".join(segment.get("text", "").strip() for segment in reviewed_segments).strip()
        accepted = bool(reviewed_segments) and (
            not original_segments
            or reviewed_score <= original_score + 0.05
        )

        if accepted:
            refined_segments = replace_segments_in_window(
                refined_segments,
                window.start,
                window.end,
                reviewed_segments,
            )

        review_records.append(
            {
                "index": review_idx,
                "start": round(window.start, 3),
                "end": round(window.end, 3),
                "score": round(window.score, 3),
                "reasons": list(window.reasons),
                "accepted": accepted,
                "original_score": round(original_score, 3),
                "reviewed_score": round(reviewed_score, 3),
                "before": original_text,
                "after": reviewed_text,
            }
        )
        log(
            f"[review {review_idx}/{len(review_windows)}] "
            f"{format_clock(window.start)}-{format_clock(window.end)} "
            f"{'적용' if accepted else '유지'} "
            f"(원본 {original_score:.2f} -> 리뷰 {reviewed_score:.2f})"
        )

    return refined_segments, review_records


def flatten_words(segments: Sequence[Dict]) -> List[Dict]:
    words = []
    for segment in segments:
        for word in segment.get("words", []) or []:
            words.append(dict(word))
    return words


def merge_speaker_intervals(intervals: Sequence[Dict], max_gap: float = 0.15) -> List[Dict]:
    ordered = sorted(
        (
            {
                "start": coerce_float(interval["start"]),
                "end": max(coerce_float(interval["start"]), coerce_float(interval["end"], coerce_float(interval["start"]))),
                "speaker": str(interval["speaker"]),
            }
            for interval in intervals
            if interval.get("speaker")
        ),
        key=lambda item: (item["start"], item["end"]),
    )

    merged = []
    for interval in ordered:
        if interval["end"] <= interval["start"]:
            continue
        if (
            merged
            and merged[-1]["speaker"] == interval["speaker"]
            and interval["start"] <= merged[-1]["end"] + max_gap
        ):
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(dict(interval))
    return merged


def trim_intervals_to_window(intervals: Sequence[Dict], window_start: float, window_end: float) -> List[Dict]:
    trimmed = []
    for interval in intervals:
        start = max(interval["start"], window_start)
        end = min(interval["end"], window_end)
        if end <= start:
            continue
        trimmed.append(
            {
                "start": start,
                "end": end,
                "speaker": interval["speaker"],
            }
        )
    return merge_speaker_intervals(trimmed)


def extract_speaker_intervals(diarized_payload: Dict) -> List[Dict]:
    normalized = normalize_result(diarized_payload)
    intervals = []

    for segment in normalized["segments"]:
        used_word_level = False
        for word in segment.get("words", []) or []:
            speaker = word.get("speaker") or segment.get("speaker")
            if not speaker:
                continue
            intervals.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "speaker": speaker,
                }
            )
            used_word_level = True

        if not used_word_level and segment.get("speaker"):
            intervals.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                }
            )

    return merge_speaker_intervals(intervals)


def extend_merged_intervals(
    merged: List[Dict],
    new_intervals: Sequence[Dict],
    max_gap: float = 0.15,
) -> List[Dict]:
    for interval in new_intervals:
        if not merged:
            merged.append(dict(interval))
            continue

        previous = merged[-1]
        if (
            previous["speaker"] == interval["speaker"]
            and interval["start"] <= previous["end"] + max_gap
        ):
            previous["end"] = max(previous["end"], interval["end"])
        else:
            merged.append(dict(interval))
    return merged


def choose_nearest_speaker(
    query_start: float,
    query_end: float,
    previous_interval: Optional[Dict],
    next_interval: Optional[Dict],
) -> Optional[str]:
    query_mid = midpoint(query_start, query_end)
    candidates = []

    if previous_interval is not None:
        candidates.append((abs(query_mid - previous_interval["end"]), previous_interval["speaker"]))
    if next_interval is not None:
        candidates.append((abs(next_interval["start"] - query_mid), next_interval["speaker"]))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def assign_speaker_for_span(
    start: float,
    end: float,
    intervals: Sequence[Dict],
    interval_idx: int,
    previous_interval: Optional[Dict],
) -> Tuple[str, int, Optional[Dict]]:
    span_end = max(start, end)

    while interval_idx < len(intervals) and intervals[interval_idx]["end"] <= start:
        previous_interval = intervals[interval_idx]
        interval_idx += 1

    best_speaker = None
    best_overlap = 0.0
    scan_idx = interval_idx
    while scan_idx < len(intervals) and intervals[scan_idx]["start"] < span_end:
        overlap = max(
            0.0,
            min(span_end, intervals[scan_idx]["end"]) - max(start, intervals[scan_idx]["start"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = intervals[scan_idx]["speaker"]
        scan_idx += 1

    if best_speaker:
        return best_speaker, interval_idx, previous_interval

    next_interval = intervals[interval_idx] if interval_idx < len(intervals) else None
    nearest = choose_nearest_speaker(start, span_end, previous_interval, next_interval)
    return nearest or UNKNOWN_SPEAKER, interval_idx, previous_interval


def dominant_speaker_from_words(words: Sequence[Dict]) -> str:
    weights: Dict[str, float] = {}
    for word in words:
        speaker = str(word.get("speaker") or UNKNOWN_SPEAKER)
        duration = max(0.05, word["end"] - word["start"])
        text_weight = max(1, len(word_text(word).strip()))
        weights[speaker] = weights.get(speaker, 0.0) + (duration * text_weight)

    if not weights:
        return UNKNOWN_SPEAKER

    known_weights = {speaker: weight for speaker, weight in weights.items() if speaker != UNKNOWN_SPEAKER}
    if known_weights:
        return max(known_weights.items(), key=lambda item: item[1])[0]
    return max(weights.items(), key=lambda item: item[1])[0]


def assign_speakers_to_segments(segments: Sequence[Dict], intervals: Sequence[Dict]) -> List[Dict]:
    if not intervals:
        output = []
        for segment in segments:
            item = dict(segment)
            if item.get("words"):
                item["words"] = [dict(word) for word in item["words"]]
                for word in item["words"]:
                    word["speaker"] = UNKNOWN_SPEAKER
            item["speaker"] = UNKNOWN_SPEAKER
            output.append(item)
        return output

    assigned = []
    interval_idx = 0
    previous_interval = None

    for segment in segments:
        item = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", ""),
        }
        if segment.get("words"):
            words = []
            for word in segment["words"]:
                labeled_word = dict(word)
                speaker, interval_idx, previous_interval = assign_speaker_for_span(
                    labeled_word["start"],
                    labeled_word["end"],
                    intervals,
                    interval_idx,
                    previous_interval,
                )
                labeled_word["speaker"] = speaker
                words.append(labeled_word)

            item["words"] = words
            item["speaker"] = dominant_speaker_from_words(words)
            item["text"] = rebuild_text_from_words(words)
            item["start"] = words[0]["start"]
            item["end"] = words[-1]["end"]
        else:
            speaker, interval_idx, previous_interval = assign_speaker_for_span(
                item["start"],
                item["end"],
                intervals,
                interval_idx,
                previous_interval,
            )
            item["speaker"] = speaker

        assigned.append(item)

    return assigned


def make_segment_from_words(words: Sequence[Dict], speaker: str) -> Dict:
    return {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": speaker or UNKNOWN_SPEAKER,
        "text": rebuild_text_from_words(words),
        "words": [dict(word) for word in words],
    }


def split_segments_by_speaker(segments: Sequence[Dict], break_gap: float) -> List[Dict]:
    output = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            item = dict(segment)
            item["speaker"] = str(item.get("speaker") or UNKNOWN_SPEAKER)
            output.append(item)
            continue

        current_words = [dict(words[0])]
        current_speaker = str(words[0].get("speaker") or segment.get("speaker") or UNKNOWN_SPEAKER)

        for word in words[1:]:
            next_word = dict(word)
            next_speaker = str(next_word.get("speaker") or current_speaker or UNKNOWN_SPEAKER)
            gap = max(0.0, next_word["start"] - current_words[-1]["end"])
            if next_speaker != current_speaker or gap > break_gap:
                output.append(make_segment_from_words(current_words, current_speaker))
                current_words = [next_word]
                current_speaker = next_speaker
            else:
                current_words.append(next_word)

        if current_words:
            output.append(make_segment_from_words(current_words, current_speaker))

    return output


def smooth_unknown_segments(segments: Sequence[Dict]) -> List[Dict]:
    smoothed = [dict(segment) for segment in segments]

    for segment in smoothed:
        if segment.get("words"):
            segment["words"] = [dict(word) for word in segment["words"]]

    for idx, segment in enumerate(smoothed):
        speaker = str(segment.get("speaker") or UNKNOWN_SPEAKER)
        if speaker != UNKNOWN_SPEAKER:
            continue

        duration = max(0.0, segment["end"] - segment["start"])
        short_text = len(segment.get("text", "").strip()) <= 12
        short_segment = duration <= 1.2 or short_text
        if not short_segment:
            continue

        prev_speaker = None
        next_speaker = None

        if idx > 0:
            prev_value = str(smoothed[idx - 1].get("speaker") or UNKNOWN_SPEAKER)
            if prev_value != UNKNOWN_SPEAKER:
                prev_speaker = prev_value
        if idx + 1 < len(smoothed):
            next_value = str(smoothed[idx + 1].get("speaker") or UNKNOWN_SPEAKER)
            if next_value != UNKNOWN_SPEAKER:
                next_speaker = next_value

        replacement = None
        if prev_speaker and prev_speaker == next_speaker:
            replacement = prev_speaker
        elif prev_speaker and not next_speaker:
            replacement = prev_speaker
        elif next_speaker and not prev_speaker:
            replacement = next_speaker

        if replacement:
            segment["speaker"] = replacement
            for word in segment.get("words", []) or []:
                word["speaker"] = replacement

    return smoothed


def merge_adjacent_segments(segments: Sequence[Dict], max_gap: float) -> List[Dict]:
    merged = []
    for segment in segments:
        item = dict(segment)
        if item.get("words"):
            item["words"] = [dict(word) for word in item["words"]]

        if not merged:
            merged.append(item)
            continue

        previous = merged[-1]
        same_speaker = str(previous.get("speaker") or "") == str(item.get("speaker") or "")
        gap = max(0.0, item["start"] - previous["end"])
        can_merge = same_speaker and gap <= max_gap

        if not can_merge:
            merged.append(item)
            continue

        previous["end"] = max(previous["end"], item["end"])

        if previous.get("words") and item.get("words"):
            previous["words"].extend(item["words"])
            previous["text"] = rebuild_text_from_words(previous["words"])
        else:
            left = previous.get("text", "").rstrip()
            right = item.get("text", "").lstrip()
            if left and right:
                previous["text"] = f"{left} {right}"
            else:
                previous["text"] = left or right

    return merged


def build_plain_segments(segments: Sequence[Dict], max_gap: float) -> List[Dict]:
    plain_segments = []
    for segment in segments:
        item = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment.get("text", "").strip(),
        }
        if segment.get("words"):
            item["words"] = [dict(word) for word in segment["words"]]
            item["text"] = rebuild_text_from_words(item["words"])
            item["start"] = item["words"][0]["start"]
            item["end"] = item["words"][-1]["end"]
        plain_segments.append(item)
    return merge_adjacent_segments(plain_segments, max_gap=max_gap)


def apply_glossary_aliases(text: str, entries: Sequence[GlossaryEntry]) -> str:
    updated = text
    replacements = []
    for entry in entries:
        for alias in entry.aliases:
            cleaned_alias = alias.strip()
            if cleaned_alias and cleaned_alias != entry.term:
                replacements.append((cleaned_alias, entry.term))
    for alias, canonical in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        updated = updated.replace(alias, canonical)
    return updated


def compress_repeated_short_tokens(text: str) -> str:
    tokens = tokenize_quality_text(text)
    if not tokens:
        return text

    compressed: List[str] = []
    changed = False
    index = 0
    while index < len(tokens):
        run_end = index + 1
        while run_end < len(tokens) and tokens[run_end] == tokens[index]:
            run_end += 1

        run_tokens = tokens[index:run_end]
        token_core = re.sub(r"[^0-9A-Za-z가-힣]+", "", tokens[index])
        if len(run_tokens) >= 4 and len(token_core) <= 2:
            compressed.extend(run_tokens[:2])
            changed = True
        else:
            compressed.extend(run_tokens)
        index = run_end

    if not changed:
        compressed = list(tokens)

    phrase_compressed: List[str] = []
    phrase_changed = False
    index = 0
    while index < len(compressed):
        if index + 1 < len(compressed):
            phrase = compressed[index : index + 2]
            phrase_core_len = sum(
                len(re.sub(r"[^0-9A-Za-z가-힣]+", "", token))
                for token in phrase
            )
            run_end = index + 2
            repeats = 1
            while run_end + 1 < len(compressed) and compressed[run_end : run_end + 2] == phrase:
                repeats += 1
                run_end += 2

            if repeats >= 4 and phrase_core_len <= 8:
                phrase_compressed.extend(phrase)
                phrase_changed = True
                index = run_end
                continue

        phrase_compressed.append(compressed[index])
        index += 1

    if not changed and not phrase_changed:
        return text
    return " ".join(phrase_compressed)


def normalize_transcript_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"([,.;:!?]){2,}", lambda match: match.group(0)[0], normalized)
    normalized = re.sub(r"\(\s+", "(", normalized)
    normalized = re.sub(r"\s+\)", ")", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def correct_transcript_text(text: str, glossary_entries: Sequence[GlossaryEntry]) -> str:
    corrected = normalize_transcript_text(text)
    corrected = apply_glossary_aliases(corrected, glossary_entries)
    corrected = compress_repeated_short_tokens(corrected)
    corrected = normalize_transcript_text(corrected)
    return corrected


def apply_text_corrections_to_segments(
    segments: Sequence[Dict],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    glossary_entries = getattr(args, "glossary_entries", [])
    if not getattr(args, "text_correction", True):
        return [clone_segment(segment) for segment in segments], []

    corrected_segments = []
    correction_records = []
    for segment in segments:
        item = clone_segment(segment)
        original_text = str(item.get("text") or "").strip()
        corrected_text = correct_transcript_text(original_text, glossary_entries)
        if corrected_text != original_text:
            item["text"] = corrected_text
            item["text_corrected"] = True
            correction_records.append(
                {
                    "start": round(item["start"], 3),
                    "end": round(item["end"], 3),
                    "before": original_text,
                    "after": corrected_text,
                }
            )
        corrected_segments.append(item)
    return corrected_segments, correction_records


def format_clock(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_subtitle_timestamp(seconds: float, kind: str) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    total_seconds, ms = divmod(milliseconds, 1000)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    separator = "," if kind == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def write_text_output(segments: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    has_speakers = any(segment.get("speaker") for segment in segments)
    with output_path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            timestamp = format_clock(segment["start"])
            text = segment.get("text", "").strip()
            if not text:
                continue
            if has_speakers and segment.get("speaker"):
                handle.write(f"[{segment['speaker']}] {timestamp} {text}\n")
            else:
                handle.write(f"{timestamp} {text}\n")


def write_subtitle_output(segments: Sequence[Dict], output_path: Path, kind: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if kind == "vtt":
            handle.write("WEBVTT\n\n")

        counter = 1
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue

            if kind == "srt":
                handle.write(f"{counter}\n")
            start = format_subtitle_timestamp(segment["start"], kind)
            end = format_subtitle_timestamp(segment["end"], kind)
            speaker_prefix = f"[{segment['speaker']}] " if segment.get("speaker") else ""
            handle.write(f"{start} --> {end}\n")
            handle.write(f"{speaker_prefix}{text}\n\n")
            counter += 1


def build_public_payload(
    audio_path: Path,
    segments: Sequence[Dict],
    language: Optional[str],
    intervals: Sequence[Dict],
    duration: float,
    chunks: Sequence[ChunkSpec],
    args: argparse.Namespace,
    review_records: Sequence[Dict],
    correction_records: Sequence[Dict],
) -> Dict:
    speaker_names = sorted(
        {
            str(segment.get("speaker"))
            for segment in segments
            if segment.get("speaker") and str(segment.get("speaker")) != UNKNOWN_SPEAKER
        }
    )
    word_count = sum(len(segment.get("words", []) or []) for segment in segments)

    return {
        "audio": str(audio_path),
        "language": language,
        "diarized": args.diarize,
        "models": {
            "transcription": effective_transcribe_model(args),
            "diarize_asr": None,
            "diarization": args.diarize_model if args.diarize else None,
        },
        "chunking": {
            "enabled": len(chunks) > 1,
            "chunk_count": len(chunks),
            "chunk_minutes": args.chunk_minutes,
            "auto_chunk_minutes": args.auto_chunk_minutes,
            "chunk_overlap_seconds": args.chunk_overlap_seconds,
        },
        "stats": {
            "duration_seconds": round(duration, 3),
            "segment_count": len(segments),
            "word_count": word_count,
            "speaker_count": len(speaker_names),
            "speakers": speaker_names,
            "speaker_interval_count": len(intervals),
        },
        "postprocess": {
            "review_pass": bool(getattr(args, "review_pass", True)),
            "review_window_count": len(review_records),
            "review_applied_count": sum(1 for item in review_records if item.get("accepted")),
            "text_correction": bool(getattr(args, "text_correction", True)),
            "text_correction_count": len(correction_records),
            "glossary_term_count": len(getattr(args, "glossary_entries", [])),
        },
        "segments": list(segments),
    }


def write_requested_outputs(
    layout: OutputLayout,
    stem: str,
    segments: Sequence[Dict],
    payload: Dict,
    formats: Sequence[str],
) -> List[Path]:
    written = []
    for fmt in formats:
        target_dir = layout.text_dir if fmt == "txt" else layout.structured_dir
        output_path = target_dir / f"{stem}.{fmt}"
        if fmt == "txt":
            write_text_output(segments, output_path)
        elif fmt == "json":
            write_json(output_path, payload)
        elif fmt in ("srt", "vtt"):
            write_subtitle_output(segments, output_path, fmt)
        else:
            continue
        written.append(output_path)
    return written


def compute_current_segments(
    transcription_segments: Sequence[Dict],
    speaker_intervals: Sequence[Dict],
    diarize: bool,
) -> List[Dict]:
    if diarize:
        labeled_segments = assign_speakers_to_segments(transcription_segments, speaker_intervals)
        split_segments = split_segments_by_speaker(
            labeled_segments,
            break_gap=DEFAULT_SEGMENT_BREAK_GAP,
        )
        smoothed_segments = smooth_unknown_segments(split_segments)
        return merge_adjacent_segments(
            smoothed_segments,
            max_gap=DEFAULT_SEGMENT_MERGE_GAP,
        )

    return build_plain_segments(
        transcription_segments,
        max_gap=min(0.35, DEFAULT_SEGMENT_MERGE_GAP),
    )


def write_progress_outputs(
    layout: OutputLayout,
    stem: str,
    formats: Sequence[str],
    audio_path: Path,
    detected_language: Optional[str],
    diarize: bool,
    args: argparse.Namespace,
    duration: float,
    chunks: Sequence[ChunkSpec],
    transcription_segments: Sequence[Dict],
    speaker_intervals: Sequence[Dict],
    review_records: Optional[Sequence[Dict]] = None,
) -> Tuple[List[Dict], Dict, List[Path]]:
    current_segments = compute_current_segments(
        transcription_segments=transcription_segments,
        speaker_intervals=speaker_intervals,
        diarize=diarize,
    )
    current_segments, correction_records = apply_text_corrections_to_segments(current_segments, args)
    payload = build_public_payload(
        audio_path=audio_path,
        segments=current_segments,
        language=detected_language,
        intervals=speaker_intervals,
        duration=duration,
        chunks=chunks,
        args=args,
        review_records=list(review_records or []),
        correction_records=correction_records,
    )
    written_outputs = write_requested_outputs(
        layout=layout,
        stem=stem,
        segments=current_segments,
        payload=payload,
        formats=formats,
    )
    return current_segments, payload, written_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mlx_whisper + whispermlx 화자 분리")
    parser.add_argument("audio", nargs="?", help="오디오 파일 경로")
    parser.add_argument(
        "--config",
        help=f"로컬 설정 JSON 경로 (기본: ./{DEFAULT_LOCAL_CONFIG_NAME})",
    )
    parser.add_argument("--diarize", action="store_true", default=None, help="화자 분리 활성화")
    parser.add_argument("--hf-token", help="HuggingFace 토큰 (미입력 시 환경변수 탐색)")
    parser.add_argument("-o", "--output-dir", help="출력 디렉토리")
    parser.add_argument(
        "--formats",
        help="출력 형식(csv): txt,json,srt,vtt 또는 all",
    )
    parser.add_argument(
        "--transcribe-model",
        default=DEFAULT_TRANSCRIBE_MODEL,
        help="mlx_whisper 전사 모델",
    )
    parser.add_argument(
        "--quality",
        choices=QUALITY_CHOICES,
        default=None,
        help=(
            f"전사 품질 프로파일 (기본: {DEFAULT_QUALITY}, "
            "max는 fp16/greedy + stricter review pass"
        ),
    )
    parser.add_argument(
        "--diarize-asr-model",
        default="small",
        help="호환성용 옵션. 현재는 사용되지 않음",
    )
    parser.add_argument(
        "--diarize-model",
        default=DEFAULT_DIARIZE_MODEL,
        help="pyannote diarization 모델",
    )
    parser.add_argument("--language", help="언어 고정 (기본: 자동 감지)")
    parser.add_argument("--initial-prompt", help="전사 첫 프롬프트")
    parser.add_argument(
        "--glossary-file",
        help="중요 용어 glossary 파일(.txt/.json)",
    )
    parser.add_argument(
        "--glossary-term",
        dest="glossary_terms",
        action="append",
        help="중요 용어를 직접 추가 (반복 가능)",
    )
    parser.add_argument(
        "--shared-glossary-file",
        help=f"공용 glossary 파일 (기본: ./{DEFAULT_SHARED_GLOSSARY_NAME})",
    )
    parser.add_argument(
        "--shared-glossary-update",
        dest="shared_glossary_update",
        action="store_true",
        default=None,
        help="완료된 전사 결과를 공용 glossary에 자동 반영",
    )
    parser.add_argument(
        "--no-shared-glossary-update",
        dest="shared_glossary_update",
        action="store_false",
        help="공용 glossary 자동 업데이트 비활성화",
    )
    parser.add_argument(
        "--review-pass",
        dest="review_pass",
        action="store_true",
        default=None,
        help="수상한 구간만 짧게 재전사하는 2차 review pass",
    )
    parser.add_argument(
        "--no-review-pass",
        dest="review_pass",
        action="store_false",
        help="2차 review pass 비활성화",
    )
    parser.add_argument(
        "--text-correction",
        dest="text_correction",
        action="store_true",
        default=None,
        help="최종 텍스트 교정 레이어 활성화",
    )
    parser.add_argument(
        "--no-text-correction",
        dest="text_correction",
        action="store_false",
        help="최종 텍스트 교정 레이어 비활성화",
    )
    parser.add_argument("--min-speakers", type=int, help="최소 화자 수")
    parser.add_argument("--max-speakers", type=int, help="최대 화자 수")
    parser.add_argument("--num-speakers", type=int, help="정확한 화자 수")
    parser.add_argument(
        "--chunk-minutes",
        type=float,
        default=DEFAULT_CHUNK_MINUTES,
        help="자동 chunking 시 chunk 길이(분)",
    )
    parser.add_argument(
        "--auto-chunk-minutes",
        type=float,
        default=DEFAULT_AUTO_CHUNK_MINUTES,
        help="이 길이(분)를 넘으면 자동 chunking",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=float,
        default=DEFAULT_CHUNK_OVERLAP_SECONDS,
        help="chunk 경계 overlap(초)",
    )
    parser.add_argument("--no-auto-chunk", action="store_true", help="자동 chunking 비활성화")
    parser.add_argument(
        "--progress-outputs",
        dest="progress_outputs",
        action="store_true",
        default=None,
        help="chunk 진행 중간 결과를 디스크에 계속 저장",
    )
    parser.add_argument(
        "--no-progress-outputs",
        dest="progress_outputs",
        action="store_false",
        help="중간 결과 저장 비활성화",
    )
    parser.add_argument("--force", action="store_true", help="cache/artifact 무시하고 재실행")
    return parser.parse_args()


def process_audio_file(
    audio_path: Path,
    layout: OutputLayout,
    args: argparse.Namespace,
    formats: Sequence[str],
    hf_token: Optional[str],
) -> None:
    layout.text_dir.mkdir(parents=True, exist_ok=True)
    layout.structured_dir.mkdir(parents=True, exist_ok=True)
    layout.artifact_root.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    artifact_dir = layout.artifact_root / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        (layout.text_dir if fmt == "txt" else layout.structured_dir) / f"{stem}.{fmt}"
        for fmt in formats
    ]
    # Word-level timestamps are only necessary when exporting structured JSON.
    # For diarized txt output, segment timestamps plus pyannote intervals are much faster.
    use_word_timestamps = "json" in formats
    log(f"\n=== 처리 시작: {audio_path.name} ===")
    log(f"전사 런타임: {describe_transcribe_runtime(args, word_timestamps=use_word_timestamps)}")
    if args.diarize and not use_word_timestamps:
        log("화자 분리 전략: pyannote interval + segment timestamp 매핑")
    log("출력 파일:")
    for expected in expected_outputs:
        log(f"- {expected}")
    log(f"- artifact: {artifact_dir}")

    work_audio_path = (
        prepare_work_audio(audio_path, artifact_dir, force=args.force)
        if args.diarize
        else audio_path
    )
    duration, chunks = plan_chunks(audio_path, artifact_dir, args)
    if len(chunks) > 1:
        log(
            f"긴 오디오 감지: {len(chunks)}개 chunk로 처리합니다 "
            f"({format_clock(duration)} 전체)."
        )

    merged_transcription_segments = []
    merged_intervals = []
    detected_language = None
    runtime_language = args.language
    written_outputs: List[Path] = []

    for chunk in chunks:
        source_audio = materialize_chunk(work_audio_path, chunk, force=args.force)
        chunk_dir = artifact_dir / f"chunk_{chunk.index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_started = time.perf_counter()
        log(
            f"[chunk {chunk.index + 1}/{len(chunks)}] "
            f"전사 중 ({format_clock(chunk.logical_start)} - {format_clock(chunk.logical_end)})..."
        )
        transcribe_started = time.perf_counter()
        mlx_result = normalize_result(
            load_or_run_mlx(
                source_audio,
                chunk_dir,
                args,
                word_timestamps=use_word_timestamps,
                language_override=runtime_language,
            )
        )
        log(
            f"[chunk {chunk.index + 1}/{len(chunks)}] 전사 완료 "
            f"({format_elapsed(time.perf_counter() - transcribe_started)})"
        )
        detected_language = detected_language or mlx_result.get("language")
        runtime_language = runtime_language or detected_language
        shifted_segments = shift_segments(mlx_result["segments"], chunk.extract_start)
        trimmed_segments = trim_segments_to_window(
            shifted_segments,
            chunk.logical_start,
            chunk.logical_end,
        )
        merged_transcription_segments.extend(trimmed_segments)

        if getattr(args, "progress_outputs", False):
            _, preview_payload, preview_outputs = write_progress_outputs(
                layout=layout,
                stem=stem,
                formats=formats,
                audio_path=audio_path,
                detected_language=detected_language,
                diarize=False,
                args=args,
                duration=duration,
                chunks=chunks,
                transcription_segments=merged_transcription_segments,
                speaker_intervals=[],
                review_records=[],
            )
            write_json(artifact_dir / "transcription.partial.json", preview_payload)
            write_json(artifact_dir / "final.partial.json", preview_payload)
            written_outputs = preview_outputs
            log(
                f"[chunk {chunk.index + 1}/{len(chunks)}] 중간 저장 완료: "
                f"{', '.join(str(path) for path in preview_outputs)} "
                f"({format_elapsed(time.perf_counter() - chunk_started)})"
            )

    review_records = []
    if merged_transcription_segments:
        refined_segments, review_records = refine_transcription_segments(
            work_audio_path,
            artifact_dir,
            args,
            merged_transcription_segments,
            duration=duration,
            word_timestamps=use_word_timestamps,
            language_override=runtime_language,
        )
        merged_transcription_segments = refined_segments
        if review_records:
            write_json(artifact_dir / "review_windows.json", {"windows": review_records})

    if args.diarize:
        log(f"화자 분리 중 (00:00 - {format_clock(duration)})...")
        diarize_started = time.perf_counter()
        merged_intervals = load_or_run_diarization(
            work_audio_path,
            artifact_dir,
            hf_token,
            args,
            model_cache_dir=shared_model_cache_dir(layout),
        )
        log(f"화자 분리 완료 ({format_elapsed(time.perf_counter() - diarize_started)})")

    merged_transcription_payload = {
        "language": detected_language,
        "segments": merged_transcription_segments,
    }
    write_json(artifact_dir / "mlx_merged.json", merged_transcription_payload)

    if args.diarize:
        write_json(artifact_dir / "speaker_intervals.json", {"intervals": merged_intervals})

    final_segments, payload, written_outputs = write_progress_outputs(
        layout=layout,
        stem=audio_path.stem,
        formats=formats,
        audio_path=audio_path,
        detected_language=detected_language,
        diarize=args.diarize,
        args=args,
        duration=duration,
        chunks=chunks,
        transcription_segments=merged_transcription_segments,
        speaker_intervals=merged_intervals,
        review_records=review_records,
    )
    write_json(artifact_dir / "final.json", payload)
    shared_glossary_path, shared_glossary_added = update_shared_glossary(
        audio_path,
        final_segments,
        args,
    )

    log("\n완료:")
    for path in written_outputs:
        log(f"- {path}")
    if shared_glossary_path and shared_glossary_added:
        log(f"- 공용 glossary 업데이트: {shared_glossary_path} (+{shared_glossary_added})")
    log(f"- artifact: {artifact_dir}")


def main() -> None:
    args = parse_args()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else Path.cwd() / DEFAULT_LOCAL_CONFIG_NAME
    )
    local_config = load_local_config(config_path)

    args.diarize = resolve_bool_option(args.diarize, local_config, "diarize", default=False)
    args.hf_token = args.hf_token or local_config.get("hf_token")
    args.formats = args.formats or local_config.get("formats") or "txt"
    args.quality = resolve_quality_option(args.quality, local_config)
    args.review_pass = resolve_bool_option(args.review_pass, local_config, "review_pass", default=True)
    args.text_correction = resolve_bool_option(
        args.text_correction,
        local_config,
        "text_correction",
        default=True,
    )
    args.language = resolve_str_option(args.language, local_config, "language")
    args.initial_prompt = resolve_str_option(args.initial_prompt, local_config, "initial_prompt")
    args.glossary_file = resolve_str_option(args.glossary_file, local_config, "glossary_file")
    args.shared_glossary_file = resolve_str_option(
        args.shared_glossary_file,
        local_config,
        "shared_glossary_file",
    )
    args.shared_glossary_path = resolve_shared_glossary_path(args.shared_glossary_file, local_config)
    args.shared_glossary_update = resolve_bool_option(
        args.shared_glossary_update,
        local_config,
        "shared_glossary_update",
        default=True,
    )
    args.glossary_entries = resolve_glossary_entries(
        args,
        local_config,
        shared_glossary_path=args.shared_glossary_path,
    )
    args.progress_outputs = resolve_bool_option(
        args.progress_outputs,
        local_config,
        "progress_outputs",
        default=DEFAULT_PROGRESS_OUTPUTS,
    )
    if args.num_speakers is None and local_config.get("num_speakers") is not None:
        args.num_speakers = int(local_config.get("num_speakers"))

    try:
        formats = parse_formats(args.formats)
        if args.num_speakers is not None and args.num_speakers < 1:
            raise RuntimeError("`num_speakers`는 1 이상이어야 합니다.")
        if args.num_speakers is not None:
            if args.min_speakers is None:
                args.min_speakers = args.num_speakers
            if args.max_speakers is None:
                args.max_speakers = args.num_speakers
        if (
            args.min_speakers is not None
            and args.max_speakers is not None
            and args.min_speakers > args.max_speakers
        ):
            raise RuntimeError("`min_speakers`는 `max_speakers`보다 클 수 없습니다.")
        audio_files = collect_audio_files(args.audio, local_config)
        output_dir = resolve_output_dir(audio_files, args.output_dir, local_config)
        layout = build_output_layout(audio_files, output_dir)
        running_matches = warn_about_other_runs()
        if running_matches:
            raise RuntimeError(
                "이미 실행 중인 전사/화자분리 프로세스가 있습니다. "
                "중복 실행은 매우 느려지니 기존 작업이 끝난 뒤 다시 실행해주세요."
            )
        acquire_run_lock(layout.artifact_root / DEFAULT_RUN_LOCK_NAME)
        run_preflight(args.diarize)

        hf_token = resolve_hf_token(args.hf_token)
        if args.diarize and not hf_token:
            raise RuntimeError(
                "--diarize 사용 시 HF 토큰이 필요합니다. "
                "--hf-token 또는 HF_TOKEN/HUGGINGFACE_TOKEN 환경변수를 사용하세요."
            )

        log(f"입력 오디오 {len(audio_files)}개를 처리합니다.")
        log(f"전사 품질 프로파일: {args.quality}")
        log(f"전사 모델: {effective_transcribe_model(args)}")
        log("전사 정밀도: fp16")
        log(f"review pass: {bool(args.review_pass)}")
        log(f"text correction: {bool(args.text_correction)}")
        log(f"progress outputs: {bool(args.progress_outputs)}")
        log(f"shared glossary update: {bool(args.shared_glossary_update)}")
        if args.glossary_entries:
            log(f"glossary 항목: {len(args.glossary_entries)}개")
        log(f"공용 glossary 파일: {args.shared_glossary_path}")
        log(f"텍스트 출력 디렉토리: {layout.text_dir}")
        if layout.structured_dir != layout.text_dir:
            log(f"구조화 출력 디렉토리: {layout.structured_dir}")
        log(f"artifact 디렉토리: {layout.artifact_root}")
        if args.diarize:
            log(f"공유 diarization 모델 캐시: {shared_model_cache_dir(layout)}")
        for audio_path in audio_files:
            args.glossary_entries = resolve_glossary_entries(
                args,
                local_config,
                shared_glossary_path=args.shared_glossary_path,
            )
            process_audio_file(
                audio_path=audio_path,
                layout=layout,
                args=args,
                formats=formats,
                hf_token=hf_token,
            )

    except Exception as exc:
        print(f"에러: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
