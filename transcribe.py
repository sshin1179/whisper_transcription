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
from functools import lru_cache
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
DEFAULT_QUALITY = "ultra"
DEFAULT_PRECISION = "fp32"
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
DEFAULT_DIARIZATION_MICRO_TURN_SECONDS = 0.35
DEFAULT_DIARIZATION_MICRO_GAP_SECONDS = 0.25
DEFAULT_SPEAKER_NEAREST_MAX_GAP = 0.75
DEFAULT_SPEAKER_CONFIDENCE_LOW = 0.35
DEFAULT_SPEAKER_CONFIDENCE_MEDIUM = 0.65
DEFAULT_OVERLAP_MIN_SECONDS = 0.12
DEFAULT_OVERLAP_MIN_SPEAKERS = 2
DEFAULT_SHORT_SPEAKER_TURN_SECONDS = 1.2
DEFAULT_SHORT_SPEAKER_TURN_TEXT_CHARS = 12
DEFAULT_SILENCE_CHUNKING = True
DEFAULT_SILENCE_THRESHOLD_DB = -35.0
DEFAULT_SILENCE_MIN_DURATION = 0.35
DEFAULT_CHUNK_BOUNDARY_SEARCH_SECONDS = 45.0
DEFAULT_CHUNK_BOUNDARY_EDGE_GUARD_SECONDS = 20.0
DEFAULT_LOCAL_CONFIG_NAME = ".transcribe.local.json"
DEFAULT_SHARED_GLOSSARY_NAME = "glossary.shared.json"
DEFAULT_ICLOUD_WHISPER_DIR = (
    Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Whisper"
)
DEFAULT_ICLOUD_INPUT_DIRNAME = "Whisper_input"
DEFAULT_ICLOUD_OUTPUT_DIRNAME = "Whisper_output"
QUALITY_CHOICES = ("fast", "normal", "max", "ultra")
PRECISION_CHOICES = ("fp16", "fp32")
DEFAULT_PROGRESS_OUTPUTS = True
DEFAULT_RUN_LOCK_NAME = ".transcribe.lock"
DEFAULT_SHARED_MODEL_CACHE_DIRNAME = "_shared_models"
DEFAULT_REVIEW_TEMPERATURES = (0.0, 0.2, 0.4)
DEFAULT_REVIEW_BEST_OF = 4
ULTRA_REVIEW_TEMPERATURES = (0.0, 0.1, 0.2, 0.4)
ULTRA_REVIEW_BEST_OF = 6
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
DEFAULT_CHUNK_CONTEXT_CHARS = 140
DEFAULT_TRANSCRIBE_SAMPLE_LEN = 128
DEFAULT_REVIEW_SAMPLE_LEN = 96
ULTRA_TRANSCRIBE_SAMPLE_LEN = 224
ULTRA_REVIEW_SAMPLE_LEN = 160
ULTRA_REVIEW_PADDING_SECONDS = 1.8
ULTRA_REVIEW_MAX_WINDOW_SECONDS = 18.0
ULTRA_REVIEW_MAX_WINDOWS = 30
ULTRA_REVIEW_MAX_TOTAL_SECONDS = 360.0
ULTRA_REVIEW_CONTEXT_CHARS = 360
ULTRA_CHUNK_CONTEXT_CHARS = 220
ULTRA_SPEAKER_COUNT_CANDIDATES = (2, 3, 4, 5, 6, 7)
DEFAULT_SPEAKER_COUNT_CLOSE_SCORE_MARGIN = 0.08
DEFAULT_HALLUCINATION_DROP_SCORE = 3.2
DEFAULT_HALLUCINATION_DROP_COMPRESSION = 8.0
DEFAULT_HALLUCINATION_DROP_NO_SPEECH = 0.72
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
SPEAKER_LABEL_STYLE_CHOICES = ("id", "name", "both")


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
class DiarizationResult:
    intervals: List[Dict]
    regular_intervals: List[Dict] = field(default_factory=list)
    exclusive_intervals: List[Dict] = field(default_factory=list)
    overlap_intervals: List[Dict] = field(default_factory=list)
    preferred: str = "intervals"
    speaker_count_profile: Dict = field(default_factory=dict)
    candidate_scores: List[Dict] = field(default_factory=list)


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
        project_data_dir() / "input",
        DEFAULT_ICLOUD_WHISPER_DIR / DEFAULT_ICLOUD_INPUT_DIRNAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def default_output_dir() -> Path:
    candidates = (
        project_data_dir() / "output",
        DEFAULT_ICLOUD_WHISPER_DIR / DEFAULT_ICLOUD_OUTPUT_DIRNAME,
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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=1)
def get_mlx_transcribe():
    try:
        from mlx_whisper.transcribe import transcribe as mlx_transcribe
    except ImportError as exc:
        raise RuntimeError(
            "mlx_whisper Python 패키지를 import하지 못했습니다. `pip install mlx-whisper` 상태를 확인하세요."
        ) from exc
    return mlx_transcribe


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


def write_text_if_changed(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except FileNotFoundError:
        pass
    path.write_text(content, encoding="utf-8")
    return True


def write_json(path: Path, payload: Dict) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    write_text_if_changed(path, content)


def is_cache_fresh(cache_path: Path, source_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        return cache_path.stat().st_mtime >= source_path.stat().st_mtime
    except FileNotFoundError:
        return False


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


def parse_int_list(value: object) -> List[int]:
    if value is None:
        return []
    raw_items: List[object]
    if isinstance(value, str):
        raw_items = [item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    parsed = []
    for item in raw_items:
        try:
            number = int(item)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"정수 목록을 읽지 못했습니다: {value}") from exc
        if number < 1:
            raise RuntimeError("정수 목록 값은 1 이상이어야 합니다.")
        if number not in parsed:
            parsed.append(number)
    return parsed


def resolve_hf_token(cli_value: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value

    for env_name in HF_TOKEN_ENV_NAMES:
        token = os.getenv(env_name)
        if token:
            return token
    return None


def ensure_hf_hub_env_token(token: Optional[str]) -> None:
    if not token:
        return
    for env_name in HF_TOKEN_ENV_NAMES:
        if not os.getenv(env_name):
            os.environ[env_name] = token


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
        return current.strip().lower()

    value = config.get("quality", DEFAULT_QUALITY)
    if not isinstance(value, str):
        raise RuntimeError("로컬 설정의 `quality` 값은 문자열이어야 합니다.")

    normalized = value.strip().lower()
    if normalized not in QUALITY_CHOICES:
        raise RuntimeError(
            "로컬 설정의 `quality` 값은 "
            + ", ".join(QUALITY_CHOICES)
            + " 중 하나여야 합니다."
        )
    return normalized


def resolve_precision_option(current: Optional[str], config: Dict) -> str:
    if current is not None:
        normalized = current.strip().lower()
    else:
        value = config.get("precision", DEFAULT_PRECISION)
        if not isinstance(value, str):
            raise RuntimeError("로컬 설정의 `precision` 값은 문자열이어야 합니다.")
        normalized = value.strip().lower()

    if normalized not in PRECISION_CHOICES:
        raise RuntimeError(
            "`precision` 값은 "
            + ", ".join(PRECISION_CHOICES)
            + " 중 하나여야 합니다."
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


def extract_participants_from_filename(audio_path: Path) -> List[str]:
    stem = re.sub(r"[_-]?\d{6,8}$", "", audio_path.stem)
    for match in re.finditer(r"\(([^()]*)\)", stem):
        participants = []
        for raw_token in match.group(1).split(","):
            token = normalize_auto_glossary_term(raw_token)
            if token:
                participants.append(token)
        if 2 <= len(participants) <= 8:
            return participants
    return []


def infer_num_speakers_from_filename(audio_path: Path) -> Optional[int]:
    participants = extract_participants_from_filename(audio_path)
    if participants:
        return len(participants)
    return None


def parse_speaker_name_assignment(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise RuntimeError("speaker 이름 매핑은 `SPEAKER_00=Name` 형식이어야 합니다.")
    speaker, name = value.split("=", 1)
    speaker = speaker.strip()
    name = name.strip()
    if not speaker or not name:
        raise RuntimeError("speaker 이름 매핑은 빈 speaker/name을 사용할 수 없습니다.")
    return speaker, name


def parse_speaker_name_map_value(value: object) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if value is None:
        return mapping
    if isinstance(value, dict):
        for speaker, name in value.items():
            speaker_id = str(speaker).strip()
            display_name = str(name).strip()
            if speaker_id and display_name:
                mapping[speaker_id] = display_name
        return mapping
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                speaker, name = parse_speaker_name_assignment(item)
                mapping[speaker] = name
            elif isinstance(item, dict):
                speaker = str(item.get("speaker") or item.get("id") or "").strip()
                name = str(item.get("name") or item.get("label") or "").strip()
                if speaker and name:
                    mapping[speaker] = name
        return mapping
    if isinstance(value, str):
        for raw_item in re.split(r"[,;\n]+", value):
            item = raw_item.strip()
            if not item:
                continue
            speaker, name = parse_speaker_name_assignment(item)
            mapping[speaker] = name
        return mapping
    raise RuntimeError("speaker 이름 매핑 형식을 읽지 못했습니다.")


def load_speaker_name_map(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            return parse_speaker_name_map_value(json.load(handle))

    mapping: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        speaker, name = parse_speaker_name_assignment(line)
        mapping[speaker] = name
    return mapping


def resolve_speaker_name_map(args: argparse.Namespace, config: Dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    config_mapping = config.get("speaker_names") or config.get("speaker_name_map")
    mapping.update(parse_speaker_name_map_value(config_mapping))

    map_file_value = args.speaker_map_file or config.get("speaker_map_file")
    if map_file_value:
        mapping.update(load_speaker_name_map(Path(str(map_file_value)).expanduser().resolve()))

    for assignment in args.speaker_names or []:
        speaker, name = parse_speaker_name_assignment(assignment)
        mapping[speaker] = name

    return mapping


def first_seen_speakers(segments: Sequence[Dict]) -> List[str]:
    seen = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "")
        if speaker and speaker != UNKNOWN_SPEAKER and speaker not in seen:
            seen.append(speaker)
        for word in segment.get("words", []) or []:
            word_speaker = str(word.get("speaker") or "")
            if word_speaker and word_speaker != UNKNOWN_SPEAKER and word_speaker not in seen:
                seen.append(word_speaker)
    return seen


def speaker_display_label(speaker: str, name: Optional[str], style: str) -> str:
    speaker = str(speaker or "").strip()
    name = str(name or "").strip()
    if not speaker:
        return name
    if speaker == UNKNOWN_SPEAKER:
        return speaker
    if style == "name" and name:
        return name
    if style == "both" and name:
        return f"{name}/{speaker}"
    return speaker


def build_speaker_display_map(
    audio_path: Path,
    segments: Sequence[Dict],
    args: argparse.Namespace,
) -> Tuple[Dict[str, str], Dict[str, bool]]:
    explicit_map = dict(getattr(args, "speaker_name_map", {}) or {})
    display_map = dict(explicit_map)
    inferred = False

    if getattr(args, "infer_speaker_names", True):
        participants = extract_participants_from_filename(audio_path)
        speakers = first_seen_speakers(segments)
        if participants and len(participants) == len(speakers):
            for speaker, participant in zip(speakers, participants):
                display_map.setdefault(speaker, participant)
            inferred = any(speaker not in explicit_map for speaker in speakers)

    return display_map, {
        "has_explicit_names": bool(explicit_map),
        "has_inferred_names": inferred,
    }


def apply_speaker_display_names(
    audio_path: Path,
    segments: Sequence[Dict],
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict]:
    display_map, source_flags = build_speaker_display_map(audio_path, segments, args)
    style = getattr(args, "speaker_label_style", "both")

    output = []
    for segment in segments:
        item = clone_segment(segment)
        speaker = str(item.get("speaker") or "")
        name = display_map.get(speaker)
        if name:
            item["speaker_name"] = name
            item["speaker_display"] = speaker_display_label(speaker, name, style)
        elif speaker:
            item["speaker_display"] = speaker_display_label(speaker, None, style)

        if item.get("words"):
            words = []
            for word in item["words"]:
                labeled_word = dict(word)
                word_speaker = str(labeled_word.get("speaker") or "")
                word_name = display_map.get(word_speaker)
                if word_name:
                    labeled_word["speaker_name"] = word_name
                    labeled_word["speaker_display"] = speaker_display_label(word_speaker, word_name, style)
                elif word_speaker:
                    labeled_word["speaker_display"] = speaker_display_label(word_speaker, None, style)
                words.append(labeled_word)
            item["words"] = words
        output.append(item)

    metadata = {
        "style": style,
        "map": display_map,
        **source_flags,
    }
    return output, metadata


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

    glossary_cache = getattr(args, "_shared_glossary_cache", None)
    if glossary_cache is None:
        glossary_cache = {}
        setattr(args, "_shared_glossary_cache", glossary_cache)

    cache_key = str(shared_path)
    existing_entries = glossary_cache.get(cache_key)
    if existing_entries is None:
        existing_entries = dedupe_glossary_entries(load_glossary_file(shared_path, required=False))
        glossary_cache[cache_key] = existing_entries

    merged_entries = dedupe_glossary_entries(
        existing_entries + build_shared_glossary_seed_entries(audio_path, segments, args)
    )

    existing_signature = [(entry.term, entry.aliases) for entry in existing_entries]
    merged_signature = [(entry.term, entry.aliases) for entry in merged_entries]
    if merged_signature == existing_signature:
        return shared_path, 0

    write_glossary_file(shared_path, merged_entries)
    glossary_cache[cache_key] = merged_entries
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


def effective_transcribe_model(args: argparse.Namespace, *, review_pass: bool = False) -> str:
    _ = review_pass
    if args.transcribe_model != DEFAULT_TRANSCRIBE_MODEL:
        return args.transcribe_model
    if args.quality == "fast":
        return DEFAULT_TRANSCRIBE_MODEL
    return MAX_QUALITY_TRANSCRIBE_MODEL


def is_ultra_quality(args: argparse.Namespace) -> bool:
    return getattr(args, "quality", DEFAULT_QUALITY) == "ultra"


def is_large_quality(args: argparse.Namespace) -> bool:
    return getattr(args, "quality", DEFAULT_QUALITY) in ("max", "ultra")


def transcribe_sample_len(args: argparse.Namespace, *, review_pass: bool = False) -> int:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_SAMPLE_LEN if review_pass else ULTRA_TRANSCRIBE_SAMPLE_LEN
    return DEFAULT_REVIEW_SAMPLE_LEN if review_pass else DEFAULT_TRANSCRIBE_SAMPLE_LEN


def review_temperatures(args: argparse.Namespace) -> Tuple[float, ...]:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_TEMPERATURES
    return DEFAULT_REVIEW_TEMPERATURES


def review_best_of(args: argparse.Namespace) -> int:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_BEST_OF
    return DEFAULT_REVIEW_BEST_OF


def review_padding_seconds(args: argparse.Namespace) -> float:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_PADDING_SECONDS
    return DEFAULT_REVIEW_PADDING_SECONDS


def review_max_window_seconds(args: argparse.Namespace) -> float:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_MAX_WINDOW_SECONDS
    return DEFAULT_REVIEW_MAX_WINDOW_SECONDS


def review_max_windows(args: argparse.Namespace) -> int:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_MAX_WINDOWS
    return DEFAULT_REVIEW_MAX_WINDOWS


def review_max_total_seconds(args: argparse.Namespace) -> float:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_MAX_TOTAL_SECONDS
    return DEFAULT_REVIEW_MAX_TOTAL_SECONDS


def review_context_chars(args: argparse.Namespace) -> int:
    if is_ultra_quality(args):
        return ULTRA_REVIEW_CONTEXT_CHARS
    return DEFAULT_REVIEW_CONTEXT_CHARS


def chunk_context_chars(args: argparse.Namespace) -> int:
    if is_ultra_quality(args):
        return ULTRA_CHUNK_CONTEXT_CHARS
    return DEFAULT_CHUNK_CONTEXT_CHARS


def review_score_threshold(args: argparse.Namespace) -> float:
    if is_ultra_quality(args):
        return 0.55
    return 1.0


def review_acceptance_margin(args: argparse.Namespace) -> float:
    if is_ultra_quality(args):
        return 0.02
    return 0.05


def build_transcribe_decode_options(
    args: argparse.Namespace,
    *,
    word_timestamps: bool,
    extra_prompt: Optional[str] = None,
    review_pass: bool = False,
    language_override: Optional[str] = None,
) -> Dict[str, object]:
    is_fast_quality = args.quality == "fast"
    is_large_profile = is_large_quality(args)
    is_ultra_profile = is_ultra_quality(args)
    decode_options: Dict[str, object] = {
        "path_or_hf_repo": effective_transcribe_model(args, review_pass=review_pass),
        "verbose": False,
        # Keep the first pass deterministic; short review windows can afford a fallback ladder.
        "temperature": review_temperatures(args) if (review_pass or is_ultra_profile) else 0.0,
        # Disabling previous-text conditioning reduces repetition loops on meeting audio.
        "condition_on_previous_text": False,
        # fp32 is much slower when a segment runs to Whisper's default token cap.
        # Keep dense Korean speech intact while bounding pathological decode loops.
        "sample_len": transcribe_sample_len(args, review_pass=review_pass),
        "compression_ratio_threshold": (
            2.4 if is_fast_quality else (1.75 if is_ultra_profile else (1.9 if is_large_profile else 2.0))
        ),
        "logprob_threshold": (
            -1.0 if is_fast_quality else (-0.55 if is_ultra_profile else (-0.6 if is_large_profile else -0.7))
        ),
        "no_speech_threshold": 0.65 if is_fast_quality else (0.55 if is_ultra_profile else 0.6),
        "word_timestamps": word_timestamps,
        "fp16": args.precision == "fp16",
    }
    if is_ultra_profile:
        decode_options["length_penalty"] = 1.0
        decode_options["best_of"] = review_best_of(args)

    if word_timestamps:
        decode_options["hallucination_silence_threshold"] = 1.0

    runtime_language = language_override or args.language
    if runtime_language:
        decode_options["language"] = runtime_language
    initial_prompt = compose_initial_prompt(args, extra_prompt=extra_prompt)
    if initial_prompt:
        decode_options["initial_prompt"] = initial_prompt
    if review_pass:
        decode_options["temperature"] = review_temperatures(args)
        decode_options["best_of"] = review_best_of(args)

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
    # Progress display does not affect transcription semantics; keep cache keys stable.
    decode_options["verbose"] = False
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
    details = [args.precision, "fallback-best-of" if is_ultra_quality(args) else "greedy"]
    if args.quality in ("normal", "max", "ultra"):
        details.append("no-prev-text")
    details.append(f"sample-len:{transcribe_sample_len(args)}")
    if is_ultra_quality(args):
        details.append(f"review:{review_max_windows(args)}win/{int(review_max_total_seconds(args))}s")
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

    def is_transcription_command(command: str) -> bool:
        if "transcribe.py" not in command and "diarize_segments.py" not in command:
            return False
        if "Visual Studio Code.app" in command or " --goto " in command:
            return False
        if re.search(r"(^|[/\s])caffeinate(\s|$)", command):
            return False
        return bool(
            re.search(r"(^|[/\s])python[0-9.]*\s+.*(transcribe|diarize_segments)\.py(\s|$)", command)
            or re.search(r"(^|[/\s])uv\s+.*(transcribe|diarize_segments)\.py(\s|$)", command)
        )

    matches = []
    for pid, command in process_rows:
        if pid in ancestor_pids:
            continue
        if is_transcription_command(command):
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


def detect_silence_intervals(
    audio_path: Path,
    artifact_dir: Path,
    args: argparse.Namespace,
    *,
    force: bool,
) -> List[Dict]:
    cache_path = artifact_dir / "silence_intervals.json"
    if not force and is_cache_fresh(cache_path, audio_path):
        payload = read_json(cache_path)
        return list(payload.get("silences") or [])

    ffmpeg = require_ffmpeg()
    threshold = float(getattr(args, "silence_threshold_db", DEFAULT_SILENCE_THRESHOLD_DB))
    min_duration = float(getattr(args, "silence_min_duration", DEFAULT_SILENCE_MIN_DURATION))
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(audio_path),
        "-af",
        f"silencedetect=n={threshold:.1f}dB:d={min_duration:.3f}",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("무음 구간 탐지에 실패했습니다.") from exc

    silences = []
    active_start: Optional[float] = None
    for line in completed.stderr.splitlines():
        start_match = re.search(r"silence_start:\s*([0-9.]+)", line)
        if start_match:
            active_start = float(start_match.group(1))
            continue

        end_match = re.search(
            r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)",
            line,
        )
        if end_match and active_start is not None:
            end = float(end_match.group(1))
            duration = float(end_match.group(2))
            if duration >= min_duration:
                silences.append(
                    {
                        "start": round(active_start, 3),
                        "end": round(end, 3),
                        "duration": round(duration, 3),
                    }
                )
            active_start = None

    write_json(
        cache_path,
        {
            "threshold_db": threshold,
            "min_duration": min_duration,
            "silences": silences,
        },
    )
    return silences


def choose_silence_boundary(
    silences: Sequence[Dict],
    target: float,
    *,
    lower: float,
    upper: float,
    search_seconds: float,
) -> Optional[float]:
    search_start = max(lower, target - search_seconds)
    search_end = min(upper, target + search_seconds)
    candidates = []
    for silence in silences:
        start = coerce_float(silence.get("start"))
        end = coerce_float(silence.get("end"), start)
        if end <= start:
            continue
        boundary = midpoint(start, end)
        if search_start <= boundary <= search_end:
            candidates.append(
                (
                    abs(boundary - target),
                    -min(end - start, 5.0),
                    boundary,
                )
            )
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def build_chunk_boundaries(
    duration: float,
    chunk_seconds: float,
    silences: Sequence[Dict],
    args: argparse.Namespace,
) -> List[float]:
    search_seconds = max(0.0, float(getattr(args, "chunk_boundary_search_seconds", DEFAULT_CHUNK_BOUNDARY_SEARCH_SECONDS)))
    edge_guard = max(0.0, float(getattr(args, "chunk_boundary_edge_guard_seconds", DEFAULT_CHUNK_BOUNDARY_EDGE_GUARD_SECONDS)))

    boundaries = [0.0]
    current = 0.0
    while current + chunk_seconds < duration - 0.001:
        target = current + chunk_seconds
        lower = current + max(edge_guard, chunk_seconds * 0.5)
        upper = min(duration - edge_guard, current + chunk_seconds * 1.5)
        boundary = choose_silence_boundary(
            silences,
            target,
            lower=lower,
            upper=upper,
            search_seconds=search_seconds,
        )
        if boundary is None or boundary <= current + 1.0:
            boundary = target
        boundaries.append(min(duration, boundary))
        current = boundaries[-1]

    if boundaries[-1] < duration:
        boundaries.append(duration)
    return boundaries


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

    silences: List[Dict] = []
    if getattr(args, "silence_chunking", DEFAULT_SILENCE_CHUNKING):
        silences = detect_silence_intervals(audio_path, artifact_dir, args, force=args.force)
        if silences:
            log(f"무음 기반 chunk 경계 후보: {len(silences)}개")

    boundaries = (
        build_chunk_boundaries(duration, chunk_seconds, silences, args)
        if silences
        else [min(duration, idx * chunk_seconds) for idx in range(0, int(duration // chunk_seconds) + 1)]
    )
    if boundaries[-1] < duration:
        boundaries.append(duration)

    normalized_boundaries = []
    for boundary in sorted(boundary for boundary in boundaries if 0.0 <= boundary <= duration):
        if abs(boundary) <= 0.001:
            boundary = 0.0
        elif abs(boundary - duration) <= 0.001:
            boundary = duration
        else:
            boundary = round(boundary, 3)
        if normalized_boundaries and abs(boundary - normalized_boundaries[-1]) <= 0.001:
            continue
        normalized_boundaries.append(boundary)
    boundaries = normalized_boundaries
    if not boundaries or boundaries[0] != 0.0:
        boundaries.insert(0, 0.0)
    if abs(boundaries[-1] - duration) > 0.001:
        boundaries.append(duration)

    chunks = []
    for index, (logical_start, logical_end) in enumerate(zip(boundaries, boundaries[1:])):
        if logical_end <= logical_start:
            continue
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

    write_json(
        artifact_dir / "chunk_plan.json",
        {
            "duration_seconds": round(duration, 3),
            "silence_chunking": bool(silences),
            "chunk_count": len(chunks),
            "boundaries": boundaries,
            "chunks": [
                {
                    "index": chunk.index,
                    "logical_start": round(chunk.logical_start, 3),
                    "logical_end": round(chunk.logical_end, 3),
                    "extract_start": round(chunk.extract_start, 3),
                    "extract_end": round(chunk.extract_end, 3),
                }
                for chunk in chunks
            ],
        },
    )
    return duration, chunks


def materialize_chunk(audio_path: Path, chunk: ChunkSpec, force: bool) -> Path:
    if not chunk.is_chunked:
        return audio_path

    if not force and is_cache_fresh(chunk.output_audio_path, audio_path):
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
    if not force and is_cache_fresh(work_audio_path, audio_path):
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
    extra_prompt: Optional[str] = None,
    language_override: Optional[str] = None,
) -> Dict:
    output_stem = build_transcribe_cache_stem(
        args,
        word_timestamps=word_timestamps,
        extra_prompt=extra_prompt,
        language_override=language_override,
    )
    json_path = artifact_dir / "mlx" / f"{output_stem}.json"
    if not args.force and is_cache_fresh(json_path, audio_path):
        log(f"mlx cache 사용: {json_path}")
        return read_json(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_transcribe = get_mlx_transcribe()

    decode_options = build_transcribe_decode_options(
        args,
        word_timestamps=word_timestamps,
        extra_prompt=extra_prompt,
        language_override=language_override,
    )
    result = mlx_transcribe(str(audio_path), **decode_options)
    write_json(json_path, result)
    return result


def interval_speaker_count(intervals: Sequence[Dict]) -> int:
    return len(
        {
            str(interval.get("speaker"))
            for interval in intervals
            if interval.get("speaker")
        }
    )


def requested_speaker_bounds(payload: Dict) -> Tuple[Optional[int], Optional[int]]:
    requested = payload.get("requested")
    if not isinstance(requested, dict):
        return None, None

    num_speakers = requested.get("num_speakers")
    if num_speakers is not None:
        count = int(num_speakers)
        return count, count

    min_speakers = requested.get("min_speakers")
    max_speakers = requested.get("max_speakers")
    return (
        int(min_speakers) if min_speakers is not None else None,
        int(max_speakers) if max_speakers is not None else None,
    )


def speaker_count_satisfies_bounds(count: int, bounds: Tuple[Optional[int], Optional[int]]) -> bool:
    min_speakers, max_speakers = bounds
    if min_speakers is not None and count < min_speakers:
        return False
    if max_speakers is not None and count > max_speakers:
        return False
    return True


def select_preferred_diarization_intervals(payload: Dict) -> Tuple[str, List[Dict]]:
    preferred_key = str(payload.get("preferred") or "")
    if preferred_key and payload.get(preferred_key):
        preferred_intervals = list(payload.get(preferred_key, []))
        regular_intervals = list(payload.get("intervals", []))
        bounds = requested_speaker_bounds(payload)
        if (
            preferred_key != "intervals"
            and regular_intervals
            and not speaker_count_satisfies_bounds(interval_speaker_count(preferred_intervals), bounds)
            and speaker_count_satisfies_bounds(interval_speaker_count(regular_intervals), bounds)
        ):
            return "intervals", regular_intervals
        return preferred_key, preferred_intervals

    exclusive_intervals = list(payload.get("exclusive_intervals", []))
    regular_intervals = list(payload.get("intervals", []))
    if exclusive_intervals:
        bounds = requested_speaker_bounds(payload)
        if (
            regular_intervals
            and not speaker_count_satisfies_bounds(interval_speaker_count(exclusive_intervals), bounds)
            and speaker_count_satisfies_bounds(interval_speaker_count(regular_intervals), bounds)
        ):
            return "intervals", regular_intervals
        return "exclusive_intervals", exclusive_intervals
    return "intervals", regular_intervals


def speaker_set_from_intervals(intervals: Sequence[Dict]) -> List[str]:
    speakers = sorted(
        {
            str(interval.get("speaker"))
            for interval in intervals
            if interval.get("speaker")
        }
    )
    return speakers


def interval_union_duration(intervals: Sequence[Dict]) -> float:
    ranges = sorted(
        (
            coerce_float(interval.get("start")),
            coerce_float(interval.get("end"), coerce_float(interval.get("start"))),
        )
        for interval in intervals
    )
    total = 0.0
    current_start = None
    current_end = None
    for start, end in ranges:
        if end <= start:
            continue
        if current_start is None:
            current_start = start
            current_end = end
            continue
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += max(0.0, current_end - current_start)
        current_start = start
        current_end = end
    if current_start is not None:
        total += max(0.0, current_end - current_start)
    return total


def compute_speaker_durations(intervals: Sequence[Dict]) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    for interval in intervals:
        speaker = str(interval.get("speaker") or "")
        if not speaker:
            continue
        duration = max(0.0, coerce_float(interval.get("end")) - coerce_float(interval.get("start")))
        durations[speaker] = durations.get(speaker, 0.0) + duration
    return durations


def compute_overlap_intervals(intervals: Sequence[Dict]) -> List[Dict]:
    events: Dict[float, List[Tuple[str, str]]] = {}
    for interval in intervals:
        speaker = str(interval.get("speaker") or "")
        if not speaker:
            continue
        start = coerce_float(interval.get("start"))
        end = coerce_float(interval.get("end"), start)
        if end <= start:
            continue
        events.setdefault(start, []).append(("start", speaker))
        events.setdefault(end, []).append(("end", speaker))

    active_counts: Dict[str, int] = {}
    overlap_intervals = []
    last_time: Optional[float] = None
    for event_time in sorted(events):
        if last_time is not None and event_time > last_time:
            active_speakers = sorted(
                speaker
                for speaker, count in active_counts.items()
                if count > 0
            )
            if len(active_speakers) >= DEFAULT_OVERLAP_MIN_SPEAKERS:
                duration = event_time - last_time
                if duration >= DEFAULT_OVERLAP_MIN_SECONDS:
                    overlap_intervals.append(
                        {
                            "start": round(last_time, 3),
                            "end": round(event_time, 3),
                            "duration": round(duration, 3),
                            "speakers": active_speakers,
                        }
                    )

        for kind, speaker in events[event_time]:
            if kind == "end":
                active_counts[speaker] = max(0, active_counts.get(speaker, 0) - 1)
            else:
                active_counts[speaker] = active_counts.get(speaker, 0) + 1
        last_time = event_time

    return overlap_intervals


def score_diarization_candidate(intervals: Sequence[Dict], duration: float, requested_speakers: Optional[int] = None) -> Dict:
    merged = merge_speaker_intervals(intervals)
    speakers = speaker_set_from_intervals(merged)
    speaker_durations = compute_speaker_durations(merged)
    speech_seconds = interval_union_duration(merged)
    coverage = min(1.0, speech_seconds / duration) if duration > 0 else 0.0
    short_turn_count = sum(
        1
        for interval in merged
        if max(0.0, coerce_float(interval.get("end")) - coerce_float(interval.get("start"))) <= 0.7
    )
    short_turn_ratio = short_turn_count / len(merged) if merged else 1.0
    tiny_threshold = max(3.0, duration * 0.005)
    tiny_speaker_count = sum(1 for value in speaker_durations.values() if value < tiny_threshold)
    substantial_speaker_count = sum(1 for value in speaker_durations.values() if value >= tiny_threshold)
    score = coverage - (short_turn_ratio * 0.16) - (tiny_speaker_count * 0.28)
    if requested_speakers is not None and len(speakers) != requested_speakers:
        score -= abs(len(speakers) - requested_speakers) * 0.35

    return {
        "requested_speakers": requested_speakers,
        "detected_speaker_count": len(speakers),
        "substantial_speaker_count": substantial_speaker_count,
        "speakers": speakers,
        "coverage_ratio": round(coverage, 4),
        "turn_count": len(merged),
        "short_turn_count": short_turn_count,
        "short_turn_ratio": round(short_turn_ratio, 4),
        "speaker_duration_floor_seconds": round(tiny_threshold, 3),
        "tiny_speaker_count": tiny_speaker_count,
        "score": round(score, 4),
    }


def requested_count_from_candidate_score(score: Dict) -> int:
    value = score.get("requested_speakers") or score.get("detected_speaker_count") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def select_diarization_candidate(candidate_payloads: Sequence[Tuple[Dict, Dict]]) -> Tuple[Dict, Dict]:
    best_score, _ = max(candidate_payloads, key=lambda item: item[0]["score"])
    best_value = coerce_float(best_score.get("score"))
    best_short_turn_ratio = coerce_float(best_score.get("short_turn_ratio"), 1.0)

    close_candidates = [
        item
        for item in candidate_payloads
        if (
            coerce_float(item[0].get("score")) >= best_value - DEFAULT_SPEAKER_COUNT_CLOSE_SCORE_MARGIN
            and int(item[0].get("tiny_speaker_count") or 0) == 0
            and coerce_float(item[0].get("short_turn_ratio"), 1.0) <= best_short_turn_ratio + 0.08
        )
    ]
    if close_candidates:
        return max(
            close_candidates,
            key=lambda item: (
                int(item[0].get("substantial_speaker_count") or 0),
                requested_count_from_candidate_score(item[0]),
                coerce_float(item[0].get("score")),
            ),
        )

    return max(candidate_payloads, key=lambda item: item[0]["score"])


def build_speaker_count_profile(
    intervals: Sequence[Dict],
    regular_intervals: Sequence[Dict],
    duration: float,
    args: argparse.Namespace,
    *,
    preferred: str,
    candidate_scores: Optional[Sequence[Dict]] = None,
) -> Dict:
    merged = merge_speaker_intervals(intervals)
    regular_merged = merge_speaker_intervals(regular_intervals)
    speaker_durations = compute_speaker_durations(merged)
    speakers = speaker_set_from_intervals(merged)
    short_turn_count = sum(
        1
        for interval in merged
        if max(0.0, coerce_float(interval.get("end")) - coerce_float(interval.get("start"))) <= DEFAULT_SHORT_SPEAKER_TURN_SECONDS
    )
    return {
        "preferred": preferred,
        "requested": {
            "num_speakers": getattr(args, "num_speakers", None),
            "min_speakers": getattr(args, "min_speakers", None),
            "max_speakers": getattr(args, "max_speakers", None),
        },
        "detected_speaker_count": len(speakers),
        "speakers": speakers,
        "speaker_durations": {
            speaker: round(seconds, 3)
            for speaker, seconds in sorted(speaker_durations.items())
        },
        "speech_coverage_ratio": round(interval_union_duration(merged) / duration, 4) if duration > 0 else 0.0,
        "turn_count": len(merged),
        "regular_turn_count": len(regular_merged),
        "short_turn_count": short_turn_count,
        "short_turn_ratio": round(short_turn_count / len(merged), 4) if merged else 0.0,
        "candidate_scores": list(candidate_scores or []),
    }


def build_diarization_result(payload: Dict, duration: float, args: argparse.Namespace, candidate_scores: Optional[Sequence[Dict]] = None) -> DiarizationResult:
    preferred, intervals = select_preferred_diarization_intervals(payload)
    regular_intervals = list(payload.get("intervals") or [])
    exclusive_intervals = list(payload.get("exclusive_intervals") or [])
    overlap_intervals = compute_overlap_intervals(regular_intervals)
    profile = build_speaker_count_profile(
        intervals,
        regular_intervals,
        duration,
        args,
        preferred=preferred,
        candidate_scores=candidate_scores,
    )
    return DiarizationResult(
        intervals=intervals,
        regular_intervals=regular_intervals,
        exclusive_intervals=exclusive_intervals,
        overlap_intervals=overlap_intervals,
        preferred=preferred,
        speaker_count_profile=profile,
        candidate_scores=list(candidate_scores or []),
    )


def run_diarization_helper(
    audio_path: Path,
    output_json: Path,
    hf_token: str,
    args: argparse.Namespace,
    *,
    model_cache_dir: Path,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Dict:
    if not args.force and is_cache_fresh(output_json, audio_path):
        cached_payload = read_json(output_json)
        if diarization_payload_matches_request(
            cached_payload,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        ):
            log(f"diarization cache 사용: {output_json}")
            return cached_payload
        log(f"diarization cache 요청 조건 불일치, 재실행: {output_json}")

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
    if num_speakers is not None:
        cmd.extend(["--num-speakers", str(num_speakers)])
    elif min_speakers is not None:
        cmd.extend(["--min-speakers", str(min_speakers)])
    if num_speakers is None and max_speakers is not None:
        cmd.extend(["--max-speakers", str(max_speakers)])

    run_cli(cmd, "pyannote 화자 분리")
    return read_json(output_json)


def diarization_cache_path(
    output_dir: Path,
    *,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> Path:
    if num_speakers is not None:
        return output_dir / f"diarization_intervals_speakers_{num_speakers}.json"
    if min_speakers is not None or max_speakers is not None:
        min_label = str(min_speakers) if min_speakers is not None else "any"
        max_label = str(max_speakers) if max_speakers is not None else "any"
        return output_dir / f"diarization_intervals_speakers_{min_label}_{max_label}.json"
    return output_dir / "diarization_intervals.json"


def diarization_payload_matches_request(
    payload: Dict,
    *,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> bool:
    preferred, intervals = select_preferred_diarization_intervals(payload)
    _ = preferred
    if not intervals:
        return False

    requested = payload.get("requested")
    if not isinstance(requested, dict):
        return num_speakers is None and min_speakers is None and max_speakers is None

    return (
        requested.get("num_speakers") == num_speakers
        and requested.get("min_speakers") == min_speakers
        and requested.get("max_speakers") == max_speakers
    )


def load_or_run_diarization(
    audio_path: Path,
    artifact_dir: Path,
    hf_token: str,
    args: argparse.Namespace,
    *,
    model_cache_dir: Path,
    duration: float,
) -> DiarizationResult:
    output_dir = artifact_dir / "whispermlx"
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_speakers = getattr(args, "num_speakers", None)
    if exact_speakers is None and args.min_speakers is not None and args.min_speakers == args.max_speakers:
        exact_speakers = args.min_speakers

    candidate_counts = list(getattr(args, "speaker_count_candidates", []) or [])
    if (
        not candidate_counts
        and is_ultra_quality(args)
        and exact_speakers is None
        and args.min_speakers is None
        and args.max_speakers is None
    ):
        candidate_counts = list(ULTRA_SPEAKER_COUNT_CANDIDATES)
        log("ultra 화자 수 후보 자동 평가: " + ", ".join(str(item) for item in candidate_counts))
    can_run_candidates = (
        bool(candidate_counts)
        and exact_speakers is None
        and args.min_speakers is None
        and args.max_speakers is None
    )
    if can_run_candidates:
        candidate_payloads = []
        candidate_scores = []
        for count in candidate_counts:
            candidate_json = output_dir / f"diarization_intervals_speakers_{count}.json"
            payload = run_diarization_helper(
                audio_path,
                candidate_json,
                hf_token,
                args,
                model_cache_dir=model_cache_dir,
                num_speakers=count,
                min_speakers=None,
                max_speakers=None,
            )
            _, intervals = select_preferred_diarization_intervals(payload)
            score = score_diarization_candidate(intervals, duration, requested_speakers=count)
            candidate_payloads.append((score, payload))
            candidate_scores.append(score)

        selected_score, selected_payload = select_diarization_candidate(candidate_payloads)
        log(
            "화자 수 후보 선택: "
            f"{selected_score.get('requested_speakers')}명 "
            f"(score={selected_score.get('score')}, speakers={selected_score.get('detected_speaker_count')}, "
            f"substantial={selected_score.get('substantial_speaker_count')})"
        )
        return build_diarization_result(selected_payload, duration, args, candidate_scores)

    output_json = diarization_cache_path(
        output_dir,
        num_speakers=exact_speakers,
        min_speakers=None if exact_speakers is not None else args.min_speakers,
        max_speakers=None if exact_speakers is not None else args.max_speakers,
    )

    payload = run_diarization_helper(
        audio_path,
        output_json,
        hf_token,
        args,
        model_cache_dir=model_cache_dir,
        num_speakers=exact_speakers,
        min_speakers=None if exact_speakers is not None else args.min_speakers,
        max_speakers=None if exact_speakers is not None else args.max_speakers,
    )
    return build_diarization_result(payload, duration, args)


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


def quality_token_cores(tokens: Sequence[str]) -> List[str]:
    return [
        re.sub(r"[^0-9A-Za-z가-힣]+", "", token).lower()
        for token in tokens
        if re.sub(r"[^0-9A-Za-z가-힣]+", "", token)
    ]


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

    max_trigram_repeat_run = longest_adjacent_phrase_run(tokens, 3)
    if max_trigram_repeat_run >= 2:
        score += 1.8 + (0.7 * (max_trigram_repeat_run - 2))
        reasons.append(f"repeat-phrase-3:{max_trigram_repeat_run}")

    if tokens:
        short_token_ratio = sum(
            1 for token in tokens if len(re.sub(r"[^0-9A-Za-z가-힣]+", "", token)) <= 2
        ) / len(tokens)
        if len(tokens) >= 6 and short_token_ratio >= 0.7:
            score += 0.7
            reasons.append("short-token-heavy")
        token_cores = quality_token_cores(tokens)
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


def is_hallucination_like_segment(
    segment: Dict,
    *,
    score: Optional[float] = None,
    reasons: Optional[Sequence[str]] = None,
) -> bool:
    text = str(segment.get("text") or "").strip()
    tokens = tokenize_quality_text(text)
    if len(tokens) < 3:
        return False

    score = score if score is not None else score_segment_for_review(segment)[0]
    reasons = list(reasons) if reasons is not None else score_segment_for_review(segment)[1]
    reason_prefixes = {reason.split(":", 1)[0] for reason in reasons}
    token_cores = quality_token_cores(tokens)
    unique_ratio = (
        len(set(token_cores)) / len(token_cores)
        if token_cores
        else 1.0
    )
    repeat_run = longest_adjacent_token_run(tokens)
    phrase_repeat_2 = longest_adjacent_phrase_run(tokens, 2)
    phrase_repeat_3 = longest_adjacent_phrase_run(tokens, 3)
    compression_ratio = coerce_float(segment.get("compression_ratio"))
    no_speech_prob = coerce_float(segment.get("no_speech_prob"))
    repeated = (
        repeat_run >= 5
        or phrase_repeat_2 >= 4
        or phrase_repeat_3 >= 3
        or "char-repeat" in reason_prefixes
    )
    metadata_bad = (
        compression_ratio >= DEFAULT_HALLUCINATION_DROP_COMPRESSION
        or no_speech_prob >= DEFAULT_HALLUCINATION_DROP_NO_SPEECH
    )
    low_information = (
        unique_ratio <= 0.38
        or ("short-token-heavy" in reason_prefixes and len(token_cores) >= 8)
        or ("low-density" in reason_prefixes and len(token_cores) <= 12)
    )
    return score >= DEFAULT_HALLUCINATION_DROP_SCORE and repeated and (metadata_bad or low_information)


def should_accept_empty_review(
    original_segments: Sequence[Dict],
    window: ReviewWindow,
) -> bool:
    if not original_segments:
        return True

    scored_segments = [
        (segment, *score_segment_for_review(segment))
        for segment in original_segments
    ]
    if scored_segments and all(
        is_hallucination_like_segment(segment, score=score, reasons=reasons)
        for segment, score, reasons in scored_segments
    ):
        return True

    if window.start <= 1.0 and any(
        is_hallucination_like_segment(segment, score=score, reasons=reasons)
        for segment, score, reasons in scored_segments
    ):
        return True

    return False


def cap_review_window(window: ReviewWindow, duration: float, args: argparse.Namespace) -> ReviewWindow:
    span = max(0.0, window.end - window.start)
    max_window_seconds = review_max_window_seconds(args)
    if span <= max_window_seconds:
        return window
    center = window.start + (span / 2.0)
    half = max_window_seconds / 2.0
    start = max(0.0, center - half)
    end = min(duration, start + max_window_seconds)
    start = max(0.0, end - max_window_seconds)
    return ReviewWindow(
        start=start,
        end=end,
        score=window.score,
        reasons=list(window.reasons),
        source_indexes=list(window.source_indexes),
    )


def collect_review_windows(
    segments: Sequence[Dict],
    duration: float,
    args: argparse.Namespace,
    speaker_intervals: Optional[Sequence[Dict]] = None,
) -> List[ReviewWindow]:
    candidates: List[ReviewWindow] = []
    padding_seconds = review_padding_seconds(args)
    score_threshold = review_score_threshold(args)
    for idx, segment in enumerate(segments):
        score, reasons = score_segment_for_review(segment)
        if score < score_threshold:
            continue
        start = max(0.0, segment["start"] - padding_seconds)
        end = min(duration, segment["end"] + padding_seconds)
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
                args,
            )
        )

    if speaker_intervals:
        cleaned_intervals = suppress_micro_speaker_flips(speaker_intervals)
        for left, right in zip(cleaned_intervals, cleaned_intervals[1:]):
            left_speaker = str(left.get("speaker") or UNKNOWN_SPEAKER)
            right_speaker = str(right.get("speaker") or UNKNOWN_SPEAKER)
            if left_speaker == right_speaker:
                continue
            boundary = midpoint(coerce_float(left.get("end")), coerce_float(right.get("start")))
            nearby_indexes = [
                idx
                for idx, segment in enumerate(segments)
                if segment["start"] - 0.8 <= boundary <= segment["end"] + 0.8
            ]
            if not nearby_indexes:
                continue
            candidates.append(
                cap_review_window(
                    ReviewWindow(
                        start=max(0.0, boundary - padding_seconds),
                        end=min(duration, boundary + padding_seconds),
                        score=1.15,
                        reasons=[f"speaker-boundary:{left_speaker}->{right_speaker}"],
                        source_indexes=nearby_indexes[:3],
                    ),
                    duration,
                    args,
                )
            )

    if segments and segments[0]["start"] >= 8.0:
        candidates.append(
            ReviewWindow(
                start=0.0,
                end=min(duration, min(segments[0]["start"] + 2.0, review_max_window_seconds(args))),
                score=1.2,
                reasons=["leading-gap"],
                source_indexes=[],
            )
        )

    if not candidates:
        return []

    merged: List[ReviewWindow] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        if not merged or candidate.start > merged[-1].end + padding_seconds:
            merged.append(candidate)
            continue

        previous = merged[-1]
        previous.end = max(previous.end, candidate.end)
        previous.score += candidate.score
        previous.reasons.extend(reason for reason in candidate.reasons if reason not in previous.reasons)
        previous.source_indexes.extend(
            idx for idx in candidate.source_indexes if idx not in previous.source_indexes
        )
        merged[-1] = cap_review_window(previous, duration, args)

    selected: List[ReviewWindow] = []
    total_review_seconds = 0.0
    for window in sorted(merged, key=lambda item: (-item.score, item.start)):
        span = max(0.0, window.end - window.start)
        if len(selected) >= review_max_windows(args):
            break
        if selected and total_review_seconds + span > review_max_total_seconds(args):
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
    args: argparse.Namespace,
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
    return context[:review_context_chars(args)]


def build_chunk_context_prompt(segments: Sequence[Dict], args: argparse.Namespace) -> Optional[str]:
    context = " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if str(segment.get("text") or "").strip()
    )
    if not context:
        return None
    max_chars = chunk_context_chars(args)
    if len(context) <= max_chars:
        return context

    trimmed = context[-max_chars:].lstrip()
    first_space = trimmed.find(" ")
    if first_space > 0:
        trimmed = trimmed[first_space + 1 :].lstrip()
    return trimmed or context[-max_chars:]


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
    if not args.force and is_cache_fresh(json_path, audio_path):
        log(f"review cache 사용: {json_path}")
        return read_json(json_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_transcribe = get_mlx_transcribe()

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
        return 0.0
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
    speaker_intervals: Optional[Sequence[Dict]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    if not getattr(args, "review_pass", True):
        return [clone_segment(segment) for segment in segments], []

    review_windows = collect_review_windows(segments, duration, args, speaker_intervals=speaker_intervals)
    if not review_windows:
        return [clone_segment(segment) for segment in segments], []

    refined_segments = [clone_segment(segment) for segment in segments]
    review_records = []

    for review_idx, window in enumerate(review_windows, start=1):
        original_segments = collect_segments_in_window(refined_segments, window.start, window.end)
        context_prompt = build_review_context_prompt(refined_segments, window.start, window.end, args)
        review_word_timestamps = True
        reviewed_payload = normalize_result(
            load_or_run_review_window(
                audio_path,
                artifact_dir,
                args,
                window,
                word_timestamps=review_word_timestamps,
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
        accepted = (
            not reviewed_segments and should_accept_empty_review(original_segments, window)
        ) or (
            bool(reviewed_segments)
            and (
                not original_segments
                or reviewed_score <= original_score + review_acceptance_margin(args)
            )
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


def prune_hallucination_like_segments(segments: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    cleaned_segments = []
    pruned_records = []
    for segment in segments:
        score, reasons = score_segment_for_review(segment)
        if is_hallucination_like_segment(segment, score=score, reasons=reasons):
            pruned_records.append(
                {
                    "start": round(segment["start"], 3),
                    "end": round(segment["end"], 3),
                    "score": round(score, 3),
                    "reasons": list(reasons),
                    "text": str(segment.get("text") or "").strip(),
                }
            )
            continue
        cleaned_segments.append(clone_segment(segment))
    return cleaned_segments, pruned_records


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


def suppress_micro_speaker_flips(
    intervals: Sequence[Dict],
    *,
    max_duration: float = DEFAULT_DIARIZATION_MICRO_TURN_SECONDS,
    max_gap: float = DEFAULT_DIARIZATION_MICRO_GAP_SECONDS,
) -> List[Dict]:
    merged = merge_speaker_intervals(intervals)
    if len(merged) < 3:
        return merged

    changed = True
    while changed and len(merged) >= 3:
        changed = False
        output = [dict(merged[0])]
        index = 1
        while index < len(merged) - 1:
            previous = output[-1]
            current = dict(merged[index])
            following = dict(merged[index + 1])
            current_duration = max(0.0, current["end"] - current["start"])
            left_gap = max(0.0, current["start"] - previous["end"])
            right_gap = max(0.0, following["start"] - current["end"])
            if (
                current_duration <= max_duration
                and previous["speaker"] == following["speaker"]
                and left_gap <= max_gap
                and right_gap <= max_gap
            ):
                previous["end"] = max(previous["end"], following["end"])
                index += 2
                changed = True
                continue

            output.append(current)
            index += 1

        if index == len(merged) - 1:
            output.append(dict(merged[-1]))
        merged = merge_speaker_intervals(output)

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
) -> Optional[Tuple[str, float]]:
    query_mid = midpoint(query_start, query_end)
    candidates = []

    if previous_interval is not None:
        candidates.append((abs(query_mid - previous_interval["end"]), previous_interval["speaker"]))
    if next_interval is not None:
        candidates.append((abs(next_interval["start"] - query_mid), next_interval["speaker"]))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1], candidates[0][0]


def speaker_confidence_level(confidence: float) -> str:
    if confidence >= DEFAULT_SPEAKER_CONFIDENCE_MEDIUM:
        return "high"
    if confidence >= DEFAULT_SPEAKER_CONFIDENCE_LOW:
        return "medium"
    if confidence > 0:
        return "low"
    return "unknown"


def build_speaker_assignment(
    speaker: str,
    *,
    confidence: float,
    overlap_seconds: float,
    overlap_ratio: float,
    method: str,
    nearest_gap: Optional[float] = None,
) -> Dict:
    assignment = {
        "speaker": speaker or UNKNOWN_SPEAKER,
        "speaker_confidence": round(max(0.0, min(1.0, confidence)), 4),
        "speaker_confidence_level": speaker_confidence_level(confidence),
        "speaker_overlap_seconds": round(max(0.0, overlap_seconds), 4),
        "speaker_overlap_ratio": round(max(0.0, min(1.0, overlap_ratio)), 4),
        "speaker_assignment": method,
    }
    if nearest_gap is not None:
        assignment["speaker_nearest_gap"] = round(max(0.0, nearest_gap), 4)
    return assignment


def assign_speaker_for_span(
    start: float,
    end: float,
    intervals: Sequence[Dict],
    interval_idx: int,
    previous_interval: Optional[Dict],
) -> Tuple[Dict, int, Optional[Dict]]:
    span_end = max(start, end)
    span_duration = max(0.05, span_end - start)

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
        overlap_ratio = min(1.0, best_overlap / span_duration)
        return (
            build_speaker_assignment(
                best_speaker,
                confidence=overlap_ratio,
                overlap_seconds=best_overlap,
                overlap_ratio=overlap_ratio,
                method="overlap",
            ),
            interval_idx,
            previous_interval,
        )

    next_interval = intervals[interval_idx] if interval_idx < len(intervals) else None
    nearest = choose_nearest_speaker(start, span_end, previous_interval, next_interval)
    if nearest is not None:
        nearest_speaker, nearest_gap = nearest
        if nearest_gap <= DEFAULT_SPEAKER_NEAREST_MAX_GAP:
            confidence = max(0.1, 0.45 * (1.0 - (nearest_gap / DEFAULT_SPEAKER_NEAREST_MAX_GAP)))
            return (
                build_speaker_assignment(
                    nearest_speaker,
                    confidence=confidence,
                    overlap_seconds=0.0,
                    overlap_ratio=0.0,
                    method="nearest",
                    nearest_gap=nearest_gap,
                ),
                interval_idx,
                previous_interval,
            )

    nearest_gap = nearest[1] if nearest is not None else None
    return (
        build_speaker_assignment(
            UNKNOWN_SPEAKER,
            confidence=0.0,
            overlap_seconds=0.0,
            overlap_ratio=0.0,
            method="unmatched",
            nearest_gap=nearest_gap,
        ),
        interval_idx,
        previous_interval,
    )


def dominant_speaker_from_words(words: Sequence[Dict]) -> str:
    weights: Dict[str, float] = {}
    for word in words:
        speaker = str(word.get("speaker") or UNKNOWN_SPEAKER)
        duration = max(0.05, word["end"] - word["start"])
        text_weight = max(1, len(word_text(word).strip()))
        confidence = max(0.05, coerce_float(word.get("speaker_confidence"), 1.0))
        weights[speaker] = weights.get(speaker, 0.0) + (duration * text_weight * confidence)

    if not weights:
        return UNKNOWN_SPEAKER

    known_weights = {speaker: weight for speaker, weight in weights.items() if speaker != UNKNOWN_SPEAKER}
    if known_weights:
        return max(known_weights.items(), key=lambda item: item[1])[0]
    return max(weights.items(), key=lambda item: item[1])[0]


def aggregate_word_speaker_confidence(words: Sequence[Dict], speaker: str) -> Dict:
    matching = [
        word
        for word in words
        if str(word.get("speaker") or UNKNOWN_SPEAKER) == speaker
    ]
    source = matching or list(words)
    if not source:
        confidence = 0.0
        overlap_ratio = 0.0
    else:
        total_weight = 0.0
        confidence_weight = 0.0
        overlap_weight = 0.0
        for word in source:
            duration = max(0.05, coerce_float(word.get("end")) - coerce_float(word.get("start")))
            text_weight = max(1, len(word_text(word).strip()))
            weight = duration * text_weight
            total_weight += weight
            confidence_weight += coerce_float(word.get("speaker_confidence")) * weight
            overlap_weight += coerce_float(word.get("speaker_overlap_ratio")) * weight
        confidence = confidence_weight / total_weight if total_weight else 0.0
        overlap_ratio = overlap_weight / total_weight if total_weight else 0.0
    return {
        "speaker_confidence": round(confidence, 4),
        "speaker_confidence_level": speaker_confidence_level(confidence),
        "speaker_overlap_ratio": round(overlap_ratio, 4),
    }


def assign_speakers_to_segments(segments: Sequence[Dict], intervals: Sequence[Dict]) -> List[Dict]:
    if not intervals:
        output = []
        for segment in segments:
            item = dict(segment)
            if item.get("words"):
                item["words"] = [dict(word) for word in item["words"]]
                for word in item["words"]:
                    word["speaker"] = UNKNOWN_SPEAKER
                    word["speaker_confidence"] = 0.0
                    word["speaker_confidence_level"] = "unknown"
                    word["speaker_assignment"] = "no-intervals"
            item["speaker"] = UNKNOWN_SPEAKER
            item["speaker_confidence"] = 0.0
            item["speaker_confidence_level"] = "unknown"
            item["speaker_assignment"] = "no-intervals"
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
                assignment, interval_idx, previous_interval = assign_speaker_for_span(
                    labeled_word["start"],
                    labeled_word["end"],
                    intervals,
                    interval_idx,
                    previous_interval,
                )
                labeled_word.update(assignment)
                words.append(labeled_word)

            item["words"] = words
            item["speaker"] = dominant_speaker_from_words(words)
            item.update(aggregate_word_speaker_confidence(words, item["speaker"]))
            item["text"] = rebuild_text_from_words(words)
            item["start"] = words[0]["start"]
            item["end"] = words[-1]["end"]
        else:
            assignment, interval_idx, previous_interval = assign_speaker_for_span(
                item["start"],
                item["end"],
                intervals,
                interval_idx,
                previous_interval,
            )
            item.update(assignment)

        assigned.append(item)

    return assigned


def make_segment_from_words(words: Sequence[Dict], speaker: str) -> Dict:
    item = {
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "speaker": speaker or UNKNOWN_SPEAKER,
        "text": rebuild_text_from_words(words),
        "words": [dict(word) for word in words],
    }
    item.update(aggregate_word_speaker_confidence(words, item["speaker"]))
    return item


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


def smooth_unknown_segments(
    segments: Sequence[Dict],
    *,
    max_gap: float = DEFAULT_SPEAKER_NEAREST_MAX_GAP,
) -> List[Dict]:
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
            prev_gap = max(0.0, segment["start"] - smoothed[idx - 1]["end"])
            if prev_value != UNKNOWN_SPEAKER and prev_gap <= max_gap:
                prev_speaker = prev_value
        if idx + 1 < len(smoothed):
            next_value = str(smoothed[idx + 1].get("speaker") or UNKNOWN_SPEAKER)
            next_gap = max(0.0, smoothed[idx + 1]["start"] - segment["end"])
            if next_value != UNKNOWN_SPEAKER and next_gap <= max_gap:
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
            segment["speaker_assignment"] = "smoothed-unknown"
            segment["speaker_confidence"] = max(0.1, coerce_float(segment.get("speaker_confidence")))
            segment["speaker_confidence_level"] = speaker_confidence_level(segment["speaker_confidence"])
            for word in segment.get("words", []) or []:
                word["speaker"] = replacement
                word["speaker_assignment"] = "smoothed-unknown"
                word["speaker_confidence"] = max(0.1, coerce_float(word.get("speaker_confidence")))
                word["speaker_confidence_level"] = speaker_confidence_level(word["speaker_confidence"])

    return smoothed


def smooth_short_speaker_turns(
    segments: Sequence[Dict],
    *,
    max_duration: float = DEFAULT_SHORT_SPEAKER_TURN_SECONDS,
    max_text_chars: int = DEFAULT_SHORT_SPEAKER_TURN_TEXT_CHARS,
    max_gap: float = DEFAULT_SEGMENT_MERGE_GAP,
) -> List[Dict]:
    smoothed = [clone_segment(segment) for segment in segments]
    if len(smoothed) < 3:
        return smoothed

    for idx in range(1, len(smoothed) - 1):
        previous = smoothed[idx - 1]
        current = smoothed[idx]
        following = smoothed[idx + 1]

        current_speaker = str(current.get("speaker") or UNKNOWN_SPEAKER)
        previous_speaker = str(previous.get("speaker") or UNKNOWN_SPEAKER)
        following_speaker = str(following.get("speaker") or UNKNOWN_SPEAKER)
        if (
            current_speaker == UNKNOWN_SPEAKER
            or previous_speaker == UNKNOWN_SPEAKER
            or following_speaker == UNKNOWN_SPEAKER
            or previous_speaker != following_speaker
            or current_speaker == previous_speaker
        ):
            continue

        duration = max(0.0, current["end"] - current["start"])
        text_chars = len(re.sub(r"\s+", "", str(current.get("text") or "")))
        previous_duration = max(0.0, previous["end"] - previous["start"])
        following_duration = max(0.0, following["end"] - following["start"])
        left_gap = max(0.0, current["start"] - previous["end"])
        right_gap = max(0.0, following["start"] - current["end"])
        if duration > max_duration and text_chars > max_text_chars:
            continue
        if left_gap > max_gap or right_gap > max_gap:
            continue
        if previous_duration < 1.5 and following_duration < 1.5:
            continue

        current["speaker"] = previous_speaker
        current["speaker_assignment"] = "smoothed-short-turn"
        current["speaker_confidence"] = max(0.1, coerce_float(current.get("speaker_confidence")))
        current["speaker_confidence_level"] = speaker_confidence_level(current["speaker_confidence"])
        for word in current.get("words", []) or []:
            word["speaker"] = previous_speaker
            word["speaker_assignment"] = "smoothed-short-turn"
            word["speaker_confidence"] = max(0.1, coerce_float(word.get("speaker_confidence")))
            word["speaker_confidence_level"] = speaker_confidence_level(word["speaker_confidence"])

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
            previous.update(aggregate_word_speaker_confidence(previous["words"], str(previous.get("speaker") or UNKNOWN_SPEAKER)))
        else:
            left = previous.get("text", "").rstrip()
            right = item.get("text", "").lstrip()
            if left and right:
                previous["text"] = f"{left} {right}"
            else:
                previous["text"] = left or right

    return merged


def overlapping_speaker_regions(start: float, end: float, overlap_intervals: Sequence[Dict]) -> List[Dict]:
    regions = []
    for interval in overlap_intervals:
        interval_start = coerce_float(interval.get("start"))
        interval_end = coerce_float(interval.get("end"), interval_start)
        overlap = max(0.0, min(end, interval_end) - max(start, interval_start))
        if overlap <= 0:
            continue
        regions.append(
            {
                "start": interval_start,
                "end": interval_end,
                "overlap_seconds": round(overlap, 4),
                "speakers": list(interval.get("speakers") or []),
            }
        )
    return regions


def annotate_overlap_metadata(segments: Sequence[Dict], overlap_intervals: Sequence[Dict]) -> List[Dict]:
    if not overlap_intervals:
        return [clone_segment(segment) for segment in segments]

    output = []
    for segment in segments:
        item = clone_segment(segment)
        segment_regions = overlapping_speaker_regions(item["start"], item["end"], overlap_intervals)
        if segment_regions:
            speakers = sorted(
                {
                    speaker
                    for region in segment_regions
                    for speaker in region.get("speakers", [])
                }
            )
            item["overlap"] = True
            item["overlap_speakers"] = speakers
            item["overlap_seconds"] = round(sum(region["overlap_seconds"] for region in segment_regions), 4)

        if item.get("words"):
            words = []
            for word in item["words"]:
                labeled_word = dict(word)
                word_regions = overlapping_speaker_regions(
                    labeled_word["start"],
                    labeled_word["end"],
                    overlap_intervals,
                )
                if word_regions:
                    labeled_word["overlap"] = True
                    labeled_word["overlap_speakers"] = sorted(
                        {
                            speaker
                            for region in word_regions
                            for speaker in region.get("speakers", [])
                        }
                    )
                words.append(labeled_word)
            item["words"] = words
        output.append(item)
    return output


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


def append_plain_segments(
    existing_segments: Sequence[Dict],
    new_segments: Sequence[Dict],
    *,
    max_gap: float,
) -> List[Dict]:
    if not new_segments:
        return [clone_segment(segment) for segment in existing_segments]

    new_plain_segments = build_plain_segments(new_segments, max_gap=max_gap)
    if not existing_segments:
        return new_plain_segments

    merged = [clone_segment(segment) for segment in existing_segments[:-1]]
    tail_candidates = [clone_segment(existing_segments[-1])]
    tail_candidates.extend(new_plain_segments)
    merged.extend(merge_adjacent_segments(tail_candidates, max_gap=max_gap))
    return merged


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
        phrase_compressed = list(compressed)

    trigram_compressed: List[str] = []
    trigram_changed = False
    index = 0
    while index < len(phrase_compressed):
        replaced = False
        if index + 2 < len(phrase_compressed):
            phrase = phrase_compressed[index : index + 3]
            phrase_core_len = sum(
                len(re.sub(r"[^0-9A-Za-z가-힣]+", "", token))
                for token in phrase
            )
            run_end = index + 3
            repeats = 1
            while run_end + 2 < len(phrase_compressed) and phrase_compressed[run_end : run_end + 3] == phrase:
                repeats += 1
                run_end += 3
            if repeats >= 3 and phrase_core_len <= 12:
                trigram_compressed.extend(phrase)
                trigram_changed = True
                index = run_end
                replaced = True
        if replaced:
            continue
        trigram_compressed.append(phrase_compressed[index])
        index += 1

    if not changed and not phrase_changed and not trigram_changed:
        return text
    return " ".join(trigram_compressed)


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

    glossary_signature = tuple(
        (entry.term, tuple(entry.aliases))
        for entry in glossary_entries
    )
    correction_cache = getattr(args, "_text_correction_cache", None)
    if correction_cache is None:
        correction_cache = {}
        setattr(args, "_text_correction_cache", correction_cache)

    corrected_segments = []
    correction_records = []
    for segment in segments:
        item = clone_segment(segment)
        original_text = str(item.get("text") or "").strip()
        cache_key = (original_text, glossary_signature)
        corrected_text = correction_cache.get(cache_key)
        if corrected_text is None:
            corrected_text = correct_transcript_text(original_text, glossary_entries)
            correction_cache[cache_key] = corrected_text
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
    has_speakers = any(segment.get("speaker") or segment.get("speaker_display") for segment in segments)
    lines: List[str] = []
    for segment in segments:
        timestamp = format_clock(segment["start"])
        text = segment.get("text", "").strip()
        if not text:
            continue
        speaker_label = segment.get("speaker_display") or segment.get("speaker")
        if has_speakers and speaker_label:
            lines.append(f"[{speaker_label}] {timestamp} {text}")
        else:
            lines.append(f"{timestamp} {text}")
    content = "\n".join(lines)
    if lines:
        content += "\n"
    write_text_if_changed(output_path, content)


def write_subtitle_output(segments: Sequence[Dict], output_path: Path, kind: str) -> None:
    chunks: List[str] = []
    if kind == "vtt":
        chunks.append("WEBVTT\n")

    counter = 1
    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        lines: List[str] = []
        if kind == "srt":
            lines.append(str(counter))
        start = format_subtitle_timestamp(segment["start"], kind)
        end = format_subtitle_timestamp(segment["end"], kind)
        speaker_label = segment.get("speaker_display") or segment.get("speaker")
        speaker_prefix = f"[{speaker_label}] " if speaker_label else ""
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker_prefix}{text}")
        chunks.append("\n".join(lines))
        counter += 1

    content = "\n\n".join(chunks)
    if content:
        content += "\n"
    write_text_if_changed(output_path, content)


def collect_segment_words(segments: Sequence[Dict]) -> List[Dict]:
    words = []
    for segment in segments:
        words.extend(dict(word) for word in segment.get("words", []) or [])
    return words


def average_numeric(values: Sequence[float]) -> Optional[float]:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def build_quality_report(
    segments: Sequence[Dict],
    intervals: Sequence[Dict],
    overlap_intervals: Sequence[Dict],
    duration: float,
    chunks: Sequence[ChunkSpec],
    review_records: Sequence[Dict],
    correction_records: Sequence[Dict],
    hallucination_records: Sequence[Dict],
    args: argparse.Namespace,
) -> Dict:
    words = collect_segment_words(segments)
    confidence_values = [
        coerce_float(word.get("speaker_confidence"))
        for word in words
        if word.get("speaker_confidence") is not None
    ]
    if not confidence_values:
        confidence_values = [
            coerce_float(segment.get("speaker_confidence"))
            for segment in segments
            if segment.get("speaker_confidence") is not None
        ]

    low_confidence_words = [
        word
        for word in words
        if word.get("speaker_confidence") is not None
        and coerce_float(word.get("speaker_confidence")) < DEFAULT_SPEAKER_CONFIDENCE_LOW
    ]
    unknown_segments = [
        segment
        for segment in segments
        if str(segment.get("speaker") or UNKNOWN_SPEAKER) == UNKNOWN_SPEAKER
    ]
    unknown_words = [
        word
        for word in words
        if str(word.get("speaker") or UNKNOWN_SPEAKER) == UNKNOWN_SPEAKER
    ]
    speaker_turns = [
        segment
        for segment in segments
        if segment.get("speaker") and str(segment.get("speaker")) != UNKNOWN_SPEAKER
    ]
    short_turn_count = sum(
        1
        for segment in speaker_turns
        if max(0.0, coerce_float(segment.get("end")) - coerce_float(segment.get("start"))) <= DEFAULT_SHORT_SPEAKER_TURN_SECONDS
    )
    overlap_seconds = sum(
        max(0.0, coerce_float(interval.get("end")) - coerce_float(interval.get("start")))
        for interval in overlap_intervals
    )
    boundary_review_count = sum(
        1
        for record in review_records
        if any(str(reason).startswith("speaker-boundary") for reason in record.get("reasons", []))
    )

    average_confidence = average_numeric(confidence_values)
    return {
        "duration_seconds": round(duration, 3),
        "chunk_count": len(chunks),
        "silence_chunking": bool(getattr(args, "silence_chunking", DEFAULT_SILENCE_CHUNKING)),
        "segment_count": len(segments),
        "word_count": len(words),
        "speaker_interval_count": len(intervals),
        "speaker_turn_count": len(speaker_turns),
        "short_turn_count": short_turn_count,
        "short_turn_ratio": round(short_turn_count / len(speaker_turns), 4) if speaker_turns else 0.0,
        "unknown_segment_count": len(unknown_segments),
        "unknown_segment_ratio": round(len(unknown_segments) / len(segments), 4) if segments else 0.0,
        "unknown_word_count": len(unknown_words),
        "unknown_word_ratio": round(len(unknown_words) / len(words), 4) if words else 0.0,
        "low_confidence_word_count": len(low_confidence_words),
        "low_confidence_word_ratio": round(len(low_confidence_words) / len(words), 4) if words else 0.0,
        "average_speaker_confidence": round(average_confidence, 4) if average_confidence is not None else None,
        "overlap_seconds": round(overlap_seconds, 3),
        "overlap_ratio": round(overlap_seconds / duration, 4) if duration > 0 else 0.0,
        "overlap_region_count": len(overlap_intervals),
        "review_window_count": len(review_records),
        "speaker_boundary_review_count": boundary_review_count,
        "text_correction_count": len(correction_records),
        "hallucination_prune_count": len(hallucination_records),
    }


def build_public_payload(
    audio_path: Path,
    segments: Sequence[Dict],
    language: Optional[str],
    intervals: Sequence[Dict],
    overlap_intervals: Sequence[Dict],
    duration: float,
    chunks: Sequence[ChunkSpec],
    args: argparse.Namespace,
    review_records: Sequence[Dict],
    correction_records: Sequence[Dict],
    hallucination_records: Sequence[Dict],
    speaker_metadata: Optional[Dict],
    diarization_profile: Optional[Dict],
    quality_report: Optional[Dict],
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
            "diarization": args.diarize_model if args.diarize else None,
        },
        "quality": {
            "profile": args.quality,
            "precision": args.precision,
            "sample_len": transcribe_sample_len(args),
            "review_sample_len": transcribe_sample_len(args, review_pass=True),
            "review_max_windows": review_max_windows(args),
            "review_max_total_seconds": review_max_total_seconds(args),
            "review_pass": bool(getattr(args, "review_pass", True)),
            "text_correction": bool(getattr(args, "text_correction", True)),
            "word_timestamps": any(segment.get("words") for segment in segments),
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
        "speaker_labels": speaker_metadata or {},
        "diarization_profile": diarization_profile or {},
        "quality_report": quality_report or {},
        "overlap_intervals": list(overlap_intervals),
        "postprocess": {
            "review_pass": bool(getattr(args, "review_pass", True)),
            "review_window_count": len(review_records),
            "review_applied_count": sum(1 for item in review_records if item.get("accepted")),
            "text_correction": bool(getattr(args, "text_correction", True)),
            "text_correction_count": len(correction_records),
            "hallucination_prune_count": len(hallucination_records),
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
    *,
    partial: bool = False,
) -> List[Path]:
    written = []
    for fmt in formats:
        target_dir = layout.text_dir if fmt == "txt" else layout.structured_dir
        output_name = f"{stem}.partial.{fmt}" if partial else f"{stem}.{fmt}"
        output_path = target_dir / output_name
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


def remove_partial_outputs(layout: OutputLayout, stem: str, formats: Sequence[str]) -> None:
    for fmt in formats:
        target_dir = layout.text_dir if fmt == "txt" else layout.structured_dir
        partial_path = target_dir / f"{stem}.partial.{fmt}"
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass


def compute_current_segments(
    transcription_segments: Sequence[Dict],
    speaker_intervals: Sequence[Dict],
    diarize: bool,
    overlap_intervals: Optional[Sequence[Dict]] = None,
) -> List[Dict]:
    if diarize:
        cleaned_intervals = suppress_micro_speaker_flips(speaker_intervals)
        labeled_segments = assign_speakers_to_segments(transcription_segments, cleaned_intervals)
        split_segments = split_segments_by_speaker(
            labeled_segments,
            break_gap=DEFAULT_SEGMENT_BREAK_GAP,
        )
        smoothed_segments = smooth_unknown_segments(split_segments)
        smoothed_segments = smooth_short_speaker_turns(smoothed_segments)
        merged_segments = merge_adjacent_segments(
            smoothed_segments,
            max_gap=DEFAULT_SEGMENT_MERGE_GAP,
        )
        return annotate_overlap_metadata(merged_segments, overlap_intervals or [])

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
    overlap_intervals: Optional[Sequence[Dict]] = None,
    diarization_profile: Optional[Dict] = None,
    review_records: Optional[Sequence[Dict]] = None,
    hallucination_records: Optional[Sequence[Dict]] = None,
    precomputed_segments: Optional[Sequence[Dict]] = None,
    partial: bool = False,
) -> Tuple[List[Dict], Dict, List[Path]]:
    if precomputed_segments is None:
        current_segments = compute_current_segments(
            transcription_segments=transcription_segments,
            speaker_intervals=speaker_intervals,
            diarize=diarize,
            overlap_intervals=overlap_intervals or [],
        )
    else:
        current_segments = [clone_segment(segment) for segment in precomputed_segments]
    current_segments, correction_records = apply_text_corrections_to_segments(current_segments, args)
    current_segments, speaker_metadata = apply_speaker_display_names(audio_path, current_segments, args)
    quality_report = build_quality_report(
        current_segments,
        speaker_intervals,
        overlap_intervals or [],
        duration,
        chunks,
        list(review_records or []),
        correction_records,
        list(hallucination_records or []),
        args,
    )
    payload = build_public_payload(
        audio_path=audio_path,
        segments=current_segments,
        language=detected_language,
        intervals=speaker_intervals,
        overlap_intervals=overlap_intervals or [],
        duration=duration,
        chunks=chunks,
        args=args,
        review_records=list(review_records or []),
        correction_records=correction_records,
        hallucination_records=list(hallucination_records or []),
        speaker_metadata=speaker_metadata,
        diarization_profile=diarization_profile,
        quality_report=quality_report,
    )
    written_outputs = write_requested_outputs(
        layout=layout,
        stem=stem,
        segments=current_segments,
        payload=payload,
        formats=formats,
        partial=partial,
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
            "fast=turbo, normal=v3, max=v3 + stricter review pass, ultra=v3 + fallback best_of + wider review)"
        ),
    )
    parser.add_argument(
        "--precision",
        choices=PRECISION_CHOICES,
        default=None,
        help=f"전사 정밀도 (기본: {DEFAULT_PRECISION}, fp16=저메모리/고속, fp32=고품질)",
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
        "--speaker-count-candidates",
        help="화자 수 후보를 쉼표로 지정하면 후보별 pyannote를 실행해 품질 점수로 선택 (예: 2,3,4,5)",
    )
    parser.add_argument(
        "--speaker-map-file",
        help="speaker 이름 매핑 파일(JSON 또는 SPEAKER_00=Name 줄 목록)",
    )
    parser.add_argument(
        "--speaker-name",
        dest="speaker_names",
        action="append",
        help="speaker 이름 매핑 직접 추가 (예: SPEAKER_00=Shin, 반복 가능)",
    )
    parser.add_argument(
        "--infer-speaker-names",
        dest="infer_speaker_names",
        action="store_true",
        default=None,
        help="파일명 참여자 목록으로 speaker 이름을 추정",
    )
    parser.add_argument(
        "--no-infer-speaker-names",
        dest="infer_speaker_names",
        action="store_false",
        help="파일명 기반 speaker 이름 추정 비활성화",
    )
    parser.add_argument(
        "--speaker-label-style",
        choices=SPEAKER_LABEL_STYLE_CHOICES,
        default=None,
        help="출력 speaker 라벨 스타일",
    )
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
        "--silence-chunking",
        dest="silence_chunking",
        action="store_true",
        default=None,
        help="긴 오디오 chunk 경계를 무음 지점에 맞춤",
    )
    parser.add_argument(
        "--no-silence-chunking",
        dest="silence_chunking",
        action="store_false",
        help="무음 기반 chunk 경계 선택 비활성화",
    )
    parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=None,
        help=f"무음 탐지 기준 dB (기본: {DEFAULT_SILENCE_THRESHOLD_DB})",
    )
    parser.add_argument(
        "--silence-min-duration",
        type=float,
        default=None,
        help=f"무음으로 볼 최소 길이 초 (기본: {DEFAULT_SILENCE_MIN_DURATION})",
    )
    parser.add_argument(
        "--chunk-boundary-search-seconds",
        type=float,
        default=None,
        help=f"목표 chunk 경계 주변 무음 검색 반경 초 (기본: {DEFAULT_CHUNK_BOUNDARY_SEARCH_SECONDS})",
    )
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
    # Diarization needs word timestamps even for txt-only output: without them,
    # a single Whisper segment can only receive one speaker label.
    use_word_timestamps = bool(args.diarize) or "json" in formats
    log(f"\n=== 처리 시작: {audio_path.name} ===")
    log(f"전사 런타임: {describe_transcribe_runtime(args, word_timestamps=use_word_timestamps)}")
    if args.diarize:
        log("화자 분리 전략: pyannote interval + word timestamp 매핑")
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
    overlap_intervals = []
    diarization_profile: Dict = {}
    detected_language = None
    runtime_language = args.language
    written_outputs: List[Path] = []
    preview_plain_segments: List[Dict] = []

    for chunk in chunks:
        source_audio = materialize_chunk(work_audio_path, chunk, force=args.force)
        chunk_dir = artifact_dir / f"chunk_{chunk.index:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_context_prompt = build_chunk_context_prompt(merged_transcription_segments[-2:], args)

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
                extra_prompt=chunk_context_prompt,
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
            preview_plain_segments = append_plain_segments(
                preview_plain_segments,
                trimmed_segments,
                max_gap=min(0.35, DEFAULT_SEGMENT_MERGE_GAP),
            )
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
                hallucination_records=[],
                precomputed_segments=preview_plain_segments,
                partial=True,
            )
            write_json(artifact_dir / "transcription.partial.json", preview_payload)
            write_json(artifact_dir / "final.partial.json", preview_payload)
            written_outputs = preview_outputs
            log(
                f"[chunk {chunk.index + 1}/{len(chunks)}] 중간 저장 완료: "
                f"{', '.join(str(path) for path in preview_outputs)} "
                f"({format_elapsed(time.perf_counter() - chunk_started)})"
            )

    if args.diarize:
        log(f"화자 분리 중 (00:00 - {format_clock(duration)})...")
        diarize_started = time.perf_counter()
        diarization_result = load_or_run_diarization(
            work_audio_path,
            artifact_dir,
            hf_token,
            args,
            model_cache_dir=shared_model_cache_dir(layout),
            duration=duration,
        )
        merged_intervals = diarization_result.intervals
        overlap_intervals = diarization_result.overlap_intervals
        diarization_profile = diarization_result.speaker_count_profile
        log(f"화자 분리 완료 ({format_elapsed(time.perf_counter() - diarize_started)})")
        if overlap_intervals:
            overlap_seconds = sum(
                max(0.0, coerce_float(item.get("end")) - coerce_float(item.get("start")))
                for item in overlap_intervals
            )
            log(f"겹쳐 말한 구간 감지: {len(overlap_intervals)}개 / {overlap_seconds:.1f}초")

    review_records = []
    hallucination_records = []
    if merged_transcription_segments:
        refined_segments, review_records = refine_transcription_segments(
            work_audio_path,
            artifact_dir,
            args,
            merged_transcription_segments,
            duration=duration,
            word_timestamps=use_word_timestamps,
            language_override=runtime_language,
            speaker_intervals=merged_intervals if args.diarize else None,
        )
        merged_transcription_segments = refined_segments
        if review_records:
            write_json(artifact_dir / "review_windows.json", {"windows": review_records})
        merged_transcription_segments, hallucination_records = prune_hallucination_like_segments(
            merged_transcription_segments
        )
        if hallucination_records:
            write_json(
                artifact_dir / "hallucination_pruned.json",
                {"segments": hallucination_records},
            )
            log(f"hallucination 정리: {len(hallucination_records)}개 세그먼트 제거")

    merged_transcription_payload = {
        "language": detected_language,
        "segments": merged_transcription_segments,
    }
    write_json(artifact_dir / "mlx_merged.json", merged_transcription_payload)

    if args.diarize:
        write_json(
            artifact_dir / "speaker_intervals.json",
            {
                "intervals": merged_intervals,
                "overlap_intervals": overlap_intervals,
                "regular_intervals": diarization_result.regular_intervals,
                "exclusive_intervals": diarization_result.exclusive_intervals,
                "profile": diarization_profile,
            },
        )
        write_json(artifact_dir / "overlap_intervals.json", {"intervals": overlap_intervals})

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
        overlap_intervals=overlap_intervals,
        diarization_profile=diarization_profile,
        review_records=review_records,
        hallucination_records=hallucination_records,
    )
    remove_partial_outputs(layout, audio_path.stem, formats)
    write_json(artifact_dir / "final.json", payload)
    write_json(artifact_dir / "quality_report.json", payload.get("quality_report", {}))
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
    args.precision = resolve_precision_option(args.precision, local_config)
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
    args.speaker_count_candidates = parse_int_list(
        args.speaker_count_candidates
        if args.speaker_count_candidates is not None
        else local_config.get("speaker_count_candidates")
    )
    args.speaker_map_file = resolve_str_option(args.speaker_map_file, local_config, "speaker_map_file")
    args.infer_speaker_names = resolve_bool_option(
        args.infer_speaker_names,
        local_config,
        "infer_speaker_names",
        default=True,
    )
    args.speaker_label_style = (
        args.speaker_label_style
        or local_config.get("speaker_label_style")
        or "both"
    )
    if args.speaker_label_style not in SPEAKER_LABEL_STYLE_CHOICES:
        raise RuntimeError(
            "`speaker_label_style`은 "
            + ", ".join(SPEAKER_LABEL_STYLE_CHOICES)
            + " 중 하나여야 합니다."
        )
    args.speaker_name_map = resolve_speaker_name_map(args, local_config)
    args.silence_chunking = resolve_bool_option(
        args.silence_chunking,
        local_config,
        "silence_chunking",
        default=DEFAULT_SILENCE_CHUNKING,
    )
    args.silence_threshold_db = float(
        args.silence_threshold_db
        if args.silence_threshold_db is not None
        else local_config.get("silence_threshold_db", DEFAULT_SILENCE_THRESHOLD_DB)
    )
    args.silence_min_duration = float(
        args.silence_min_duration
        if args.silence_min_duration is not None
        else local_config.get("silence_min_duration", DEFAULT_SILENCE_MIN_DURATION)
    )
    args.chunk_boundary_search_seconds = float(
        args.chunk_boundary_search_seconds
        if args.chunk_boundary_search_seconds is not None
        else local_config.get("chunk_boundary_search_seconds", DEFAULT_CHUNK_BOUNDARY_SEARCH_SECONDS)
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
        if args.silence_min_duration <= 0:
            raise RuntimeError("`silence_min_duration`은 0보다 커야 합니다.")
        if args.chunk_boundary_search_seconds < 0:
            raise RuntimeError("`chunk_boundary_search_seconds`는 0 이상이어야 합니다.")
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
        ensure_hf_hub_env_token(hf_token)

        log(f"입력 오디오 {len(audio_files)}개를 처리합니다.")
        log(f"전사 품질 프로파일: {args.quality}")
        log(f"전사 모델: {effective_transcribe_model(args)}")
        log(f"전사 정밀도: {args.precision}")
        log(f"review pass: {bool(args.review_pass)}")
        log(f"text correction: {bool(args.text_correction)}")
        log(f"progress outputs: {bool(args.progress_outputs)}")
        log(f"silence chunking: {bool(args.silence_chunking)}")
        log(f"speaker label style: {args.speaker_label_style}")
        if args.speaker_name_map:
            log(f"speaker 이름 매핑: {len(args.speaker_name_map)}개")
        if args.speaker_count_candidates:
            log(f"화자 수 후보 평가: {', '.join(str(item) for item in args.speaker_count_candidates)}")
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
        base_glossary_entries = list(args.glossary_entries)
        args._shared_glossary_cache = {}
        failed_files: List[Tuple[Path, Exception]] = []
        for audio_path in audio_files:
            file_args = argparse.Namespace(**vars(args))
            file_args.glossary_entries = list(base_glossary_entries)
            try:
                if (
                    file_args.diarize
                    and file_args.num_speakers is None
                    and file_args.min_speakers is None
                    and file_args.max_speakers is None
                ):
                    inferred_speakers = infer_num_speakers_from_filename(audio_path)
                    if inferred_speakers is not None:
                        file_args.num_speakers = inferred_speakers
                        file_args.min_speakers = inferred_speakers
                        file_args.max_speakers = inferred_speakers
                        log(f"화자 수 자동 추정: {audio_path.name} -> {inferred_speakers}명")
                process_audio_file(
                    audio_path=audio_path,
                    layout=layout,
                    args=file_args,
                    formats=formats,
                    hf_token=hf_token,
                )
            except Exception as exc:
                failed_files.append((audio_path, exc))
                log(f"파일 처리 실패: {audio_path.name}: {exc}")
                if len(audio_files) == 1:
                    raise

        if failed_files:
            failed_summary = ", ".join(path.name for path, _ in failed_files)
            raise RuntimeError(f"{len(failed_files)}개 파일 처리 실패: {failed_summary}")

    except Exception as exc:
        print(f"에러: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
