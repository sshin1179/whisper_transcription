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
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/diarize_batch_${STAMP}.log"
LATEST_PATH="$LOG_DIR/latest_diarize_batch.log"
INPUT_LIST_PATH=""
TAIL_PID=""

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
  if [ -n "${TAIL_PID:-}" ]; then
    kill "$TAIL_PID" >/dev/null 2>&1 || true
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
          --output-dir "$OUTPUT_DIR" \
          --glossary-file "$JUNG_GLOSSARY_FILE" \
          --num-speakers 2
      else
        log "Optional glossary not found, continuing without it: $JUNG_GLOSSARY_FILE"
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          --output-dir "$OUTPUT_DIR" \
          --num-speakers 2
      fi
      ;;
    "Meeting (STI Jung, Shin) #2_260417.m4a")
      if [ -f "$JUNG_GLOSSARY_FILE" ]; then
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          --output-dir "$OUTPUT_DIR" \
          --glossary-file "$JUNG_GLOSSARY_FILE" \
          --language ko \
          --num-speakers 2
      else
        log "Optional glossary not found, continuing without it: $JUNG_GLOSSARY_FILE"
        run_job "$label" \
          python3 transcribe.py "$audio_path" \
          --diarize --quality max --formats txt --progress-outputs \
          --output-dir "$OUTPUT_DIR" \
          --language ko \
          --num-speakers 2
      fi
      ;;
    "Meeting (BNK Gwak, STI Shin)_260417.m4a")
      run_job "$label" \
        python3 transcribe.py "$audio_path" \
        --diarize --quality max --formats txt --progress-outputs \
        --output-dir "$OUTPUT_DIR" \
        --language ko \
        --num-speakers 2
      ;;
    "Meeting (Hana Yu, STI Jo).m4a")
      run_job "$label" \
        python3 transcribe.py "$audio_path" \
        --diarize --quality max --formats txt --progress-outputs \
        --output-dir "$OUTPUT_DIR" \
        --language ko \
        --num-speakers 3
      ;;
    *)
      run_job "$label" \
        python3 transcribe.py "$audio_path" \
        --diarize --quality max --formats txt --progress-outputs \
        --output-dir "$OUTPUT_DIR" \
        --language ko
      ;;
  esac
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
  log "Artifacts stay local at: $LOCAL_DATA_DIR/artifacts"

  if [ ! -d "$INPUT_DIR" ]; then
    log "Input directory not found: $INPUT_DIR"
    exit 1
  fi

  INPUT_LIST_PATH="$(mktemp "${TMPDIR:-/tmp}/transcribe_inputs.XXXXXX")"
  find "$INPUT_DIR" -maxdepth 1 -type f \
    \( -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.wav' -o -iname '*.flac' -o -iname '*.aac' -o -iname '*.mp4' -o -iname '*.mov' \) \
    | sort > "$INPUT_LIST_PATH"

  input_count="$(wc -l < "$INPUT_LIST_PATH" | tr -d '[:space:]')"
  if [ "$input_count" = "0" ]; then
    log "No audio files found in $INPUT_DIR"
    exit 0
  fi

  log "Queued ${input_count} input files from $INPUT_DIR"

  while IFS= read -r audio_path <&3; do
    [ -n "$audio_path" ] || continue
    audio_name="$(basename "$audio_path")"
    label="${audio_name%.*}"
    run_audio_job "$audio_path" "$audio_name" "$label"
  done 3< "$INPUT_LIST_PATH"

  log "Detached diarize batch finished"
} >> "$LOG_PATH" 2>&1
