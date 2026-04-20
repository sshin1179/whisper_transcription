# Whisper Transcription (Mac, Apple Silicon)

Apple Silicon Mac에서 미팅 녹음을 로컬로 전사하는 도구.
Notta 같은 유료 서비스 없이, 더 높은 퀄리티로 전사 + 화자 분리까지 가능.

최근 업데이트:
- 기본 전사 품질 프로파일을 `max + fp32 + greedy`로 두고, review pass만 더 공격적으로 보정
- 수상한 구간만 짧게 다시 보는 selective review pass 추가
- glossary 기반 전사 보정 + 최종 텍스트 교정 레이어 추가
- `mlx_whisper` word timestamp 기반 후처리
- `pyannote` diarization만 직접 호출해서 불필요한 ASR 중복 제거
- diarize는 chunk마다 반복하지 않고 파일 단위로 한 번만 실행
- `pyannote`의 exclusive diarization을 우선 사용해 STT/화자 매칭 품질 개선
- `whispermlx` 화자 분리 결과를 더 촘촘하게 병합
- diarize 시 공유 16kHz mono WAV를 한 번만 만들어 전사/화자분리 재사용
- txt-only 전사에서는 불필요한 word timestamp 생성을 생략
- 중간 progress 출력은 기본적으로 켜서 긴 파일 처리 상황을 확인 가능
- `txt/json/srt/vtt` 출력 지원
- artifact/cache 저장으로 재실행 속도 개선
- 긴 오디오 자동 chunking
- HF 토큰 환경변수 지원

## How it works

두 개의 오픈소스 도구를 조합해서 각각의 장점만 사용:

| 단계 | 도구 | 역할 |
|------|------|------|
| 전사 | [mlx-whisper](https://github.com/ml-explore/mlx-examples) | Metal GPU 가속, hallucination 방지 옵션 |
| 화자 분리 | [whispermlx](https://github.com/kalebjs/whispermlx) | pyannote 기반 speaker diarization |

> **왜 이렇게?**
> whispermlx 단독 사용 시 한국어에서 hallucination("Jelly Jelly..." 반복)이 발생.
> mlx-whisper는 `--hallucination-silence-threshold` 옵션으로 이를 방지할 수 있지만 화자 분리가 없음.
> 그래서 **전사는 mlx-whisper, 화자 분리는 whispermlx**에서 가져와 병합.

## Setup

### 1. mlx-whisper 설치

```bash
pip3 install mlx-whisper
```

설치 후 확인:
```bash
mlx_whisper --help
```

### 2. whispermlx 설치 (화자 분리 사용 시)

Python 3.10+ 필요. [uv](https://github.com/astral-sh/uv)로 설치하는 게 가장 간단:

```bash
# uv 없으면 먼저 설치
brew install uv

# whispermlx 설치
uv tool install whispermlx --python 3.12
```

설치 후 확인:
```bash
whispermlx --help
```

### 3. HuggingFace 토큰 발급 (화자 분리 사용 시)

1. https://huggingface.co/settings/tokens 에서 토큰 발급
2. 아래 두 모델 페이지에서 "Agree and access repository" 클릭:
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

## Usage

### 전사만 (화자 분리 없이)

```bash
python3 transcribe.py meeting.m4a
```

기본적으로 `--quality max`, `fp32`, `greedy`로 실행되며, 별도 모델을 지정하지 않았을 때는
`mlx-community/whisper-large-v3-mlx`를 사용한다. 또한 반복 루프를 줄이기 위해
이전 텍스트 conditioning은 끄고, review pass에서만 더 넓은 fallback 재시도(`best_of`)를 사용한다.
추가로 기본적으로 review pass와 text correction이 켜져 있어, 수상한 구간만 짧게 재전사하고
최종 출력 텍스트를 glossary/후처리 규칙으로 한 번 더 다듬는다.

로컬 설정 파일(`.transcribe.local.json`)에 기본 오디오/토큰을 넣어두면:

```bash
python3 transcribe.py
```

출력: `./meeting.txt`

```
00:00 네. 저희 그래서 30일에 먼저 태국으로 가세요?
00:15 스미님은?
00:17 예정상으로는 그렇긴 합니다.
```

artifact도 함께 생성:

- `./meeting.artifacts/final.json`
- `./meeting.artifacts/mlx_merged.json`

### 전사 + 화자 분리

토큰은 CLI 인자 또는 환경변수로 전달 가능:

```bash
export HF_TOKEN=hf_your_token
python3 transcribe.py meeting.m4a --diarize
```

또는:

```bash
python3 transcribe.py meeting.m4a --diarize --hf-token hf_your_token
```

또는 로컬 전용 설정 파일:

```json
{
  "audio": "./recordings/meeting.m4a",
  "diarize": true,
  "quality": "max",
  "hf_token": "hf_your_token",
  "output_dir": ".",
  "formats": "txt"
}
```

파일명은 `./.transcribe.local.json`이며 `.gitignore`에 포함해 두는 것을 권장.

출력: `./meeting.txt`

```
[SPEAKER_03] 00:00 네. 저희 그래서 30일에 먼저 태국으로 가세요?
[SPEAKER_03] 00:15 스미님은?
[SPEAKER_02] 00:17 예정상으로는 그렇긴 합니다.
```

화자 수를 알고 있으면 정확도와 속도를 같이 챙길 수 있다:

```bash
python3 transcribe.py meeting.m4a --diarize --num-speakers 2
```

### 출력 디렉토리 지정

```bash
python3 transcribe.py meeting.m4a --diarize -o ./output
```

### 배치 실행

Input 폴더의 새 파일을 최신 항목부터 순서대로 전사하려면:

```bash
bash ./scripts/run_detached_diarize_batch.sh
```

기본 흐름은 아래와 같다.

- `Whisper_output/<파일명>.txt`가 이미 있고 비어 있지 않으면 새 파일 큐에서는 건너뜀
- 새 파일은 생성시간/수정시간/파일명 날짜 기준으로 최신 항목부터 처리

기존 결과/cache까지 무시하고 전부 fp32로 다시 돌릴 때:

```bash
WHISPER_BATCH_FORCE_RERUN=1 bash ./scripts/run_detached_diarize_batch.sh
```

삭제된 `txt`만 artifact에서 복구하고 싶을 때는 명시적으로 켠다:

```bash
WHISPER_RESTORE_FROM_ARTIFACTS=1 bash ./scripts/run_detached_diarize_batch.sh
```

### 여러 출력 형식 생성

```bash
python3 transcribe.py meeting.m4a --formats txt,json,srt,vtt
```

또는:

```bash
python3 transcribe.py meeting.m4a --formats all
```

### 품질 프로파일

기본값은 `max + fp32 + greedy`다.

```bash
# 기본값: 최고 품질(v3)
python3 transcribe.py meeting.m4a

# 속도 우선(turbo)
python3 transcribe.py meeting.m4a --quality fast

# 균형형(v3)
python3 transcribe.py meeting.m4a --quality normal
```

`normal`과 `max`에서는 커스텀 `--transcribe-model`을 따로 주지 않았을 때
`mlx-community/whisper-large-v3-mlx`를 사용하며, 1차 전사는 기본적으로 `fp32 + greedy`로 실행한다.
`fast`는 `mlx-community/whisper-large-v3-turbo`를 사용한다.
수상한 구간만 review pass에서 `best_of + temperature fallback`으로 다시 본다.
반복 방지를 위해 previous-text conditioning도 끈 상태다.

### 품질 보정 레이어

기본적으로 아래 3단계가 함께 동작한다.

- selective review pass: `avg_logprob`, `compression_ratio`, 반복 패턴 등을 보고 수상한 구간만 짧게 재전사
- glossary prompt: 중요한 사람 이름/회사명/약어를 초기 프롬프트에 실어 전사 품질 보정
- text correction: glossary alias 치환, 공백/구두점 정리, 짧은 반복 토큰 압축

끄고 싶으면:

```bash
python3 transcribe.py meeting.m4a --no-review-pass --no-text-correction
```

glossary를 직접 넣고 싶으면:

```bash
python3 transcribe.py meeting.m4a \
  --glossary-term 고디바 \
  --glossary-term 직지랩스 \
  --glossary-term 시공테크
```

또는 파일로:

```text
# one term per line
고디바
직지랩스|직지 랩스
시공테크|시공 테크
```

```bash
python3 transcribe.py meeting.m4a --glossary-file ./glossary.txt
```

### 긴 오디오 자동 chunking

기본적으로 60분이 넘는 오디오는 30분 단위로 자동 분할해서 처리한다.

```bash
python3 transcribe.py long_meeting.m4a
```

직접 조정:

```bash
python3 transcribe.py long_meeting.m4a --chunk-minutes 20 --auto-chunk-minutes 40
```

자동 chunking 비활성화:

```bash
python3 transcribe.py long_meeting.m4a --no-auto-chunk
```

### 화자 분리 속도

현재 화자 분리 단계는 불필요한 ASR/align를 생략하고 `pyannote` diarization만 직접 실행한다.
또한 diarize 모드에서는 공유 작업용 WAV를 한 번만 만들어 전사/화자분리에서 같이 사용한다.
긴 오디오여도 diarization pipeline은 chunk마다 다시 띄우지 않고 파일 단위로 한 번만 실행한다.
가능한 경우 pyannote의 `exclusive` diarization 결과를 우선 사용해 word-speaker reconciliation 오차를 줄인다.

전사만 할 때 `--formats txt`로 실행하면 word timestamp를 만들지 않아 더 빠르다.
읽어보는 용도의 1차 검토는 `txt`만 먼저 뽑는 것이 가장 효율적이다.
`--formats json`에서만 word-level 데이터가 유지된다.
JSON이 필요할 때만 `--formats txt,json`처럼 명시해서 추가하면 된다.

기본적으로 chunk 중간 결과를 계속 저장한다. 실행 중간 저장을 끄고 싶을 때만:

```bash
python3 transcribe.py meeting.m4a --no-progress-outputs
```

## Tips

### `--language` 옵션 사용하지 않기

자동 감지가 훨씬 정확. 한국어+영어 섞인 미팅에서 언어를 고정하면 오히려 품질이 떨어짐.

### 첫 실행은 느림

모델을 다운로드해야 해서 첫 실행에 시간이 걸림 (약 3GB). 이후에는 캐시되어 빠르게 동작.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.9+ (mlx-whisper용)
- [uv](https://github.com/astral-sh/uv) (whispermlx 설치용)
- ffmpeg: `brew install ffmpeg`

## Output

기본 출력:

- `meeting.txt`

선택 출력:

- `meeting.json`
- `meeting.srt`
- `meeting.vtt`

artifact/cache:

- `meeting.artifacts/final.json`: 최종 병합 결과
- `meeting.artifacts/mlx_merged.json`: mlx 전사 병합본
- `meeting.artifacts/speaker_intervals.json`: 화자 구간
- `meeting.artifacts/chunk_*/...`: chunk별 raw 결과

## Benchmarks

20분 한국어 미팅 기준:

| | whisper_transcription | Notta (유료) |
|---|---|---|
| **전사 정확도** | 높음 | 보통 |
| **Hallucination** | 없음 | 없음 |
| **화자 분리** | 있음 (짧은 추임새 간혹 흔들림) | 있음 (안정적) |
| **고유명사/약어** | 정확 | 누락/오인식 많음 |
| **문장 분리** | 한 줄씩 깔끔 | 긴 덩어리 |
| **한국어** | 강함 | 보통 |
| **영어** | 강함 | 강함 |
| **비용** | 무료 (로컬) | 유료 |
