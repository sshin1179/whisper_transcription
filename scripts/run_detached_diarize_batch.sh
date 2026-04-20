#!/usr/bin/env bash

set -eu
if (set -o pipefail) >/dev/null 2>&1; then
  set -o pipefail
fi

ROOT="/Users/seunghyeonshin/whisper_transcription"
ICLOUD_ROOT="${WHISPER_ICLOUD_ROOT:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/Whisper}"
INPUT_DIR="${WHISPER_INPUT_DIR:-$ICLOUD_ROOT/Whisper_input}"
OUTPUT_DIR="${WHISPER_OUTPUT_DIR:-$ICLOUD_ROOT/Whisper_output}"
LOCAL_DATA_DIR="$ROOT/Data"
LOG_DIR="$LOCAL_DATA_DIR/artifacts/_batch_logs"
JUNG_GLOSSARY_FILE="$ROOT/glossary.jung1.txt"
BATCH_LOCK_DIR="$LOG_DIR/run_detached_diarize_batch.lock"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/diarize_batch_${STAMP}.log"
LATEST_PATH="$LOG_DIR/latest_diarize_batch.log"
INPUT_LIST_PATH=""
PENDING_LIST_PATH=""
TAIL_PID=""
BATCH_LOCK_OWNED="0"
BATCH_FORCE_RERUN="${WHISPER_BATCH_FORCE_RERUN:-0}"
RESTORE_FROM_ARTIFACTS="${WHISPER_RESTORE_FROM_ARTIFACTS:-0}"
FORCE_ARG=""
if [ "$BATCH_FORCE_RERUN" = "1" ]; then
  FORCE_ARG="--force"
fi

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

if command -v caffeinate >/dev/null 2>&1; then
  CAFFEINATE_BIN="$(command -v caffeinate)"
else
  CAFFEINATE_BIN=""
fi

cleanup() {
  if [ -n "${INPUT_LIST_PATH:-}" ] && [ -f "$INPUT_LIST_PATH" ]; then
    rm -f "$INPUT_LIST_PATH"
  fi
  if [ -n "${PENDING_LIST_PATH:-}" ] && [ -f "$PENDING_LIST_PATH" ]; then
    rm -f "$PENDING_LIST_PATH"
  fi
  if [ -n "${TAIL_PID:-}" ]; then
    kill "$TAIL_PID" >/dev/null 2>&1 || true
  fi
  if [ "${BATCH_LOCK_OWNED:-0}" = "1" ] && [ -d "$BATCH_LOCK_DIR" ]; then
    lock_pid=""
    if [ -f "$BATCH_LOCK_DIR/pid" ]; then
      lock_pid="$(tr -d '[:space:]' < "$BATCH_LOCK_DIR/pid" 2>/dev/null || true)"
    fi
    if [ -z "$lock_pid" ] || [ "$lock_pid" = "$$" ]; then
      rm -rf "$BATCH_LOCK_DIR"
    fi
  fi
}

trap cleanup EXIT INT TERM

mkdir -p "$LOCAL_DATA_DIR" "$LOG_DIR" "$INPUT_DIR" "$OUTPUT_DIR"
ln -sfn "$INPUT_DIR" "$LOCAL_DATA_DIR/input"
ln -sfn "$OUTPUT_DIR" "$LOCAL_DATA_DIR/output"
: > "$LOG_PATH"
ln -sfn "$(basename "$LOG_PATH")" "$LATEST_PATH"

if [ -t 1 ]; then
  printf 'Streaming batch log to terminal: %s\n' "$LOG_PATH"
  tail -n 0 -f "$LOG_PATH" &
  TAIL_PID="$!"
fi

cd "$ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

is_pid_running() {
  pid="$1"
  if [ -z "$pid" ] || [ "$pid" -le 0 ] 2>/dev/null; then
    return 1
  fi
  kill -0 "$pid" >/dev/null 2>&1
}

build_ancestor_pid_list() {
  pid="$1"
  ancestors=" $pid "
  while true; do
    parent_pid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    if [ -z "$parent_pid" ] || [ "$parent_pid" -le 1 ] 2>/dev/null; then
      break
    fi
    case "$ancestors" in
      *" $parent_pid "*) break ;;
    esac
    ancestors="${ancestors}${parent_pid} "
    pid="$parent_pid"
  done
  printf '%s\n' "$ancestors"
}

list_competing_runs() {
  ancestor_pids="$(build_ancestor_pid_list "$$")"
  ps -axo pid=,ppid=,command= | awk \
    -v ancestors="$ancestor_pids" '
      {
        pid = $1
        $1 = ""
        $2 = ""
        sub(/^  */, "", $0)
        cmd = $0
        if (index(ancestors, " " pid " ") > 0) {
          next
        }
        if ((index(cmd, "transcribe.py") > 0) || (index(cmd, "diarize_segments.py") > 0)) {
          print pid "\t" cmd
        }
      }
    '
}

acquire_batch_lock() {
  poll_seconds="${WHISPER_WAIT_POLL_SECONDS:-30}"
  while true; do
    if mkdir "$BATCH_LOCK_DIR" >/dev/null 2>&1; then
      printf '%s\n' "$$" > "$BATCH_LOCK_DIR/pid"
      BATCH_LOCK_OWNED="1"
      return 0
    fi

    existing_pid=""
    if [ -f "$BATCH_LOCK_DIR/pid" ]; then
      existing_pid="$(tr -d '[:space:]' < "$BATCH_LOCK_DIR/pid" 2>/dev/null || true)"
    fi
    if is_pid_running "$existing_pid"; then
      log "다른 batch 스크립트가 실행 중입니다. 종료될 때까지 ${poll_seconds}초 간격으로 대기합니다. (PID ${existing_pid})"
      sleep "$poll_seconds"
      continue
    fi

    rm -rf "$BATCH_LOCK_DIR"
  done
}

wait_for_other_runs() {
  wait_enabled="${WHISPER_WAIT_FOR_OTHER_RUNS:-1}"
  poll_seconds="${WHISPER_WAIT_POLL_SECONDS:-30}"
  [ "$wait_enabled" = "0" ] && return 0

  while true; do
    matches="$(list_competing_runs)"
    if [ -z "$matches" ]; then
      return 0
    fi

    log "이미 실행 중인 전사/화자분리 작업이 있어 현재 batch는 대기합니다."
    printf '%s\n' "$matches" | head -n 5 | while IFS="$(printf '\t')" read -r pid command; do
      [ -n "$pid" ] || continue
      log "- PID ${pid}: ${command}"
    done
    sleep "$poll_seconds"
  done
}

run_with_sleep_prevention() {
  if [ -n "$CAFFEINATE_BIN" ]; then
    "$CAFFEINATE_BIN" -i -m "$@"
  else
    "$@"
  fi
}

run_job() {
  label="$1"
  shift

  log "START ${label}"
  run_with_sleep_prevention "$@" </dev/null
  log "DONE ${label}"
}

run_audio_job() {
  audio_path="$1"
  audio_name="$2"
  label="$3"

  case "$audio_name" in
    "Meeting (STI Jung, Shin) #1_260417.m4a")
      if [ -f "$JUNG_GLOSSARY_FILE" ]; then
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          ${FORCE_ARG:+$FORCE_ARG} \
          --output-dir "$OUTPUT_DIR" \
          --glossary-file "$JUNG_GLOSSARY_FILE" \
          --language ko \
          --num-speakers 2
      else
        log "Optional glossary not found, continuing without it: $JUNG_GLOSSARY_FILE"
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          ${FORCE_ARG:+$FORCE_ARG} \
          --output-dir "$OUTPUT_DIR" \
          --language ko \
          --num-speakers 2
      fi
      ;;
    "Meeting (STI Jung, Shin) #2_260417.m4a")
      if [ -f "$JUNG_GLOSSARY_FILE" ]; then
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          ${FORCE_ARG:+$FORCE_ARG} \
          --output-dir "$OUTPUT_DIR" \
          --glossary-file "$JUNG_GLOSSARY_FILE" \
          --language ko \
          --num-speakers 2
      else
        log "Optional glossary not found, continuing without it: $JUNG_GLOSSARY_FILE"
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          ${FORCE_ARG:+$FORCE_ARG} \
          --output-dir "$OUTPUT_DIR" \
          --language ko \
          --num-speakers 2
      fi
      ;;
    "Meeting (BNK Gwak, STI Shin)_260417.m4a")
      run_job "$label" \
        python3 transcribe.py "$audio_path" \
        --diarize --quality max --formats txt --progress-outputs \
        ${FORCE_ARG:+$FORCE_ARG} \
        --output-dir "$OUTPUT_DIR" \
        --language ko \
        --num-speakers 2
      ;;
    "Meeting (Hana Yu, STI Jo).m4a"|"Meeting (Hana Yu, STI Jo)_260416.m4a")
      run_job "$label" \
        python3 transcribe.py "$audio_path" \
        --diarize --quality max --formats txt --progress-outputs \
        ${FORCE_ARG:+$FORCE_ARG} \
        --output-dir "$OUTPUT_DIR" \
        --language ko \
        --num-speakers 3
      ;;
    *)
      run_job "$label" \
        python3 transcribe.py "$audio_path" \
        --diarize --quality max --formats txt --progress-outputs \
        ${FORCE_ARG:+$FORCE_ARG} \
        --output-dir "$OUTPUT_DIR" \
        --language ko
      ;;
  esac
}

expected_output_path() {
  audio_path="$1"
  audio_name="$(basename "$audio_path")"
  printf '%s/%s.txt\n' "$OUTPUT_DIR" "${audio_name%.*}"
}

artifact_final_path() {
  audio_path="$1"
  audio_name="$(basename "$audio_path")"
  printf '%s/artifacts/%s/final.json\n' "$LOCAL_DATA_DIR" "${audio_name%.*}"
}

restore_txt_from_final_artifact() {
  final_path="$1"
  output_path="$2"
  [ -s "$final_path" ] || return 1

  python3 - "$final_path" "$output_path" <<'PY'
import json
import sys
from pathlib import Path


def format_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


final_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
payload = json.loads(final_path.read_text(encoding="utf-8"))
segments = payload.get("segments") or []
if not segments:
    raise SystemExit(2)

output_path.parent.mkdir(parents=True, exist_ok=True)
temp_path = output_path.with_name(output_path.name + ".tmp")
has_speakers = any(segment.get("speaker") for segment in segments)
written = 0
with temp_path.open("w", encoding="utf-8") as handle:
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        timestamp = format_clock(float(segment.get("start") or 0.0))
        speaker = segment.get("speaker")
        if has_speakers and speaker:
            handle.write(f"[{speaker}] {timestamp} {text}\n")
        else:
            handle.write(f"{timestamp} {text}\n")
        written += 1

if written == 0:
    temp_path.unlink(missing_ok=True)
    raise SystemExit(2)
temp_path.replace(output_path)
PY
}

{
  log "Detached diarize batch started"
  log "Log path: $LOG_PATH"
  if [ -n "$CAFFEINATE_BIN" ]; then
    log "Sleep prevention: enabled via caffeinate -i -m (display sleep is allowed; closing the lid can still suspend the Mac)"
  else
    log "Sleep prevention: caffeinate not found; running without explicit sleep prevention"
  fi
  log "Input directory: $INPUT_DIR"
  log "Output directory: $OUTPUT_DIR"
  log "Transcribe quality: max"
  log "Progress outputs: enabled"
  if [ "$BATCH_FORCE_RERUN" = "1" ]; then
    log "Full rerun: enabled (--force, existing txt/cache/artifacts ignored)"
  else
    log "Full rerun: disabled"
  fi
  if [ "$RESTORE_FROM_ARTIFACTS" = "1" ]; then
    log "Artifact txt restore: enabled"
  else
    log "Artifact txt restore: disabled"
  fi
  log "Artifacts stay local at: $LOCAL_DATA_DIR/artifacts"
  acquire_batch_lock
  wait_for_other_runs

  if [ ! -d "$INPUT_DIR" ]; then
    log "Input directory not found: $INPUT_DIR"
    exit 1
  fi

  INPUT_LIST_PATH="$(mktemp "${TMPDIR:-/tmp}/transcribe_inputs.XXXXXX")"
  PENDING_LIST_PATH="$(mktemp "${TMPDIR:-/tmp}/transcribe_pending_inputs.XXXXXX")"
  python3 - "$INPUT_DIR" > "$INPUT_LIST_PATH" <<'PY'
from pathlib import Path
import re
import sys

input_dir = Path(sys.argv[1]).expanduser()
audio_extensions = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".mov", ".wav", ".webm"}


def filename_date_key(path: Path) -> int:
    matches = re.findall(r"\d{6,8}", path.stem)
    return int(matches[-1]) if matches else 0


def sort_key(path: Path) -> tuple:
    stat = path.stat()
    created_at = getattr(stat, "st_birthtime", stat.st_mtime)
    return (
        -created_at,
        -stat.st_mtime,
        -filename_date_key(path),
        path.name.lower(),
    )


audio_files = [
    path
    for path in input_dir.iterdir()
    if path.is_file() and path.suffix.lower() in audio_extensions
]
for path in sorted(audio_files, key=sort_key):
    print(path)
PY

  input_count="$(wc -l < "$INPUT_LIST_PATH" | tr -d '[:space:]')"
  if [ "$input_count" = "0" ]; then
    log "No audio files found in $INPUT_DIR"
    exit 0
  fi

  skip_count=0
  queued_count=0
  while IFS= read -r audio_path; do
    [ -n "$audio_path" ] || continue
    output_path="$(expected_output_path "$audio_path")"
    if [ "$BATCH_FORCE_RERUN" != "1" ] && [ -s "$output_path" ]; then
      log "SKIP $(basename "$audio_path") (existing output: $output_path)"
      skip_count=$((skip_count + 1))
      continue
    fi
    final_path="$(artifact_final_path "$audio_path")"
    if [ "$BATCH_FORCE_RERUN" != "1" ] && [ "$RESTORE_FROM_ARTIFACTS" = "1" ] && [ -s "$final_path" ] && [ "$final_path" -nt "$audio_path" ]; then
      if restore_txt_from_final_artifact "$final_path" "$output_path"; then
        log "RESTORE $(basename "$audio_path") (fresh artifact: $final_path -> $output_path)"
        skip_count=$((skip_count + 1))
        continue
      fi
      log "Artifact restore failed for $(basename "$audio_path"), queueing transcription"
    fi
    printf '%s\n' "$audio_path" >> "$PENDING_LIST_PATH"
    queued_count=$((queued_count + 1))
  done < "$INPUT_LIST_PATH"

  log "Found ${input_count} input files from $INPUT_DIR"
  if [ "$skip_count" -gt 0 ]; then
    log "Skipping ${skip_count} files with existing txt outputs in $OUTPUT_DIR"
  fi
  if [ "$queued_count" = "0" ]; then
    log "No new audio files to transcribe"
  else
    log "Queued ${queued_count} new input files from $INPUT_DIR"

    while IFS= read -r audio_path <&3; do
      [ -n "$audio_path" ] || continue
      audio_name="$(basename "$audio_path")"
      label="${audio_name%.*}"
      run_audio_job "$audio_path" "$audio_name" "$label"
    done 3< "$PENDING_LIST_PATH"
  fi

  log "Detached diarize batch finished"
} >> "$LOG_PATH" 2>&1
