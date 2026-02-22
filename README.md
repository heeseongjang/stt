# STT (Speech-to-Text) with faster-whisper

faster-whisper를 사용한 음성 인식(STT) 테스트 프로젝트

## 프로젝트 구조

```
stt/
├── main.py                  # 실행 진입점 (CLI)
├── src/
│   ├── __init__.py
│   ├── transcriber.py       # Transcriber 클래스 (모델 로드 + 변환)
│   └── corrector.py         # Ollama 로컬 LLM 후처리 교정
├── tests/
│   ├── __init__.py
│   └── test_transcriber.py
├── audio/                   # 오디오 파일 + 프롬프트 파일(.txt) 보관
├── requirements.txt
└── .gitignore
```

## 설치

```bash
# ffmpeg 필요 (macOS)
brew install ffmpeg

# 가상환경 생성 (Python 3.12 기준)
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 사용법

```bash
# 기본 실행 (언어 자동 감지)
python main.py audio/EI_0001.mp4

# 언어 지정 (한국어) - 언어를 알면 지정하는 게 정확도 10~15% 향상
python main.py audio/EI_0001.mp4 ko
```

### 프롬프트 파일

오디오 파일과 같은 이름의 `.txt` 파일을 만들면 자동으로 `initial_prompt`로 사용됩니다.

```
audio/
├── EI_0001.mp4
├── EI_0001.txt    ← "한국어 뉴스 보도입니다."
├── EI_0002.mp4
└── EI_0002.txt    ← "한국어 인터뷰입니다."
```

- `.txt` 파일이 없으면 기본값 `"한국어 음성입니다."` 사용
- **짧고 일반적인 맥락**만 적는 게 좋음
- 구체적인 단어를 나열하면 오히려 역효과 (모델이 모든 구간에서 해당 단어를 우선 선택하려 함)

## 지원 모델

### 기본 모델 (다국어 지원, `.en` 붙으면 영어 전용)

| 모델 | 파라미터 | 크기 | 특징 |
|------|----------|------|------|
| tiny | 39M | ~150MB | 가장 빠름, 정확도 낮음. 리소스 제한 환경용 |
| base | 74M | ~300MB | 빠른 속도, 낮은 정확도. 간단한 테스트용 |
| small | 244M | ~1GB | 속도/정확도 균형. 일반 용도 추천 |
| medium | 769M | ~3GB | 높은 정확도. 프로덕션 품질 필요 시 |
| large-v3 | 1.5B | ~6GB | 최고 정확도. 전문 전사/다국어에 최적 |

### 경량화 모델 (distilled - 원본 대비 50% 작고 6배 빠름, 정확도 1% 이내 손실)

| 모델 | 특징 |
|------|------|
| distil-small.en | small 경량화, 영어 전용 |
| distil-medium.en | medium 경량화, 영어 전용 |
| distil-large-v2 | large-v2 경량화, 다국어 |
| distil-large-v3 | large-v3 경량화, 다국어. 가성비 최고 |

### 터보 모델 (디코더 32층→4층 축소, large-v3 대비 6배 빠름)

| 모델 | 파라미터 | 크기 | 특징 |
|------|----------|------|------|
| large-v3-turbo | 809M | ~3GB | large-v3급 정확도 + 빠른 속도. 실시간용 추천 |

## 주요 옵션 설명

### language (언어 지정)

| 상황 | 추천 |
|------|------|
| 언어가 확실할 때 | `language="ko"` (정확도 10~15% 향상) |
| 어떤 언어인지 모를 때 | `language=None` (자동 감지) |
| 잘못된 언어를 넣으면 | None보다 훨씬 나쁨. 영어에 ko 넣으면 "Hello"가 "헬로"로 나옴 |

### initial_prompt (힌트 텍스트)

- 모델에 맥락 힌트를 주는 텍스트
- 짧고 일반적인 맥락만 주는 게 좋음 (예: `"한국어 뉴스 보도입니다."`)
- 구체적인 단어를 나열하면 역효과 — 모델이 모든 구간에서 해당 단어를 우선 선택하려 해서 다른 구간까지 망가짐

### VAD (Voice Activity Detection)

무음 구간을 자동으로 건너뛰는 기능. 인터뷰 같은 작은 음성이 누락되면 조정 필요.

| 파라미터 | 기본값 | 현재값 | 설명 |
|---------|--------|--------|------|
| threshold | 0.5 | 0.3 | 음성 감지 민감도. 낮을수록 작은 소리도 잡음 |
| min_silence_duration_ms | 2000 | 300 | 이 시간 이상 무음이면 문장 끊김. 낮추면 짧은 쉼도 연결 |
| speech_pad_ms | 30 | 200 | 음성 앞뒤 여유 시간. 높이면 첫/끝 음절 잘림 방지 |

### 디코딩 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| beam_size | 5 | 동시 탐색 후보 수. 높이면 정확도↑ 속도↓. 테스트 결과 5와 10 차이 미미 |
| temperature | [0.0~1.0] | 단어 선택의 모험도. 0.0=확실한 것만, 1.0=다양하게. 단계적 폴백 방식 |
| repetition_penalty | 1.0 | 이미 나온 단어의 확률을 깎는 값. 높이면 정상 반복도 바뀔 수 있음 |
| no_repeat_ngram_size | 0 | N개 단어 조합 반복 시 완전 차단. 0=비활성화 |
| condition_on_previous_text | True | 이전 세그먼트 결과를 다음 인식에 참고 |

### 필터링 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| no_speech_threshold | 0.6 | 무음 판단 기준. 높이면 배경음 필터 강화 |
| log_prob_threshold | -1.0 | 인식 확신도 기준. 낮추면 확신 낮은 인식도 허용 |
| compression_ratio_threshold | 2.4 | 반복 텍스트 감지. gzip 압축률이 이 값 넘으면 재시도 |

### temperature 폴백 흐름

```
세그먼트 인식 (temperature=0.0)
  ↓
압축률 > 2.4? 또는 확신도 < -1.0?
  ├─ NO  → 결과 채택
  └─ YES → temperature=0.2로 재시도
             ↓
           또 실패? → temperature=0.4로 재시도
             ↓
           ... 최대 1.0까지 시도
```

## 테스트 결과 & 교훈

### 효과가 있었던 것
- `initial_prompt="한국어 뉴스 보도입니다."` — 물음표 임의 추가 해결, 전반적 인식률 향상
- VAD 파라미터 조정 (threshold=0.3) — 누락되던 인터뷰 구간 인식 성공
- `large-v3` 모델 사용 — base 대비 확연한 정확도 차이

### 효과가 없거나 역효과였던 것
- `initial_prompt`에 구체적 단어 나열 ("박판, 보닛, 충돌 시") — 다른 구간까지 오염
- `hotwords="박판"` — "바꾸므로"가 "핫꿈으로"로 오히려 악화
- 여러 옵션 동시 변경 (beam_size=10 + repetition_penalty=1.2 + ...) — 정확도 하락
- `beam_size` 5→10 단독 변경 — 거의 차이 없음, 속도만 느려짐

### 모델 한계 (faster-whisper로 해결 불가)
- 전문 용어 오인식: "합금으로" → "하품으로"
- 단어 누락: "알루미늄 합금으로" → "알루미늄으로"
- 유사 발음 오인식: "충돌 때" → "충돌대", "충격을" → "상의를"
- 이런 경우 LLM 후처리 교정이 필요 (`src/corrector.py`)

## 후처리 교정 (Ollama 로컬 LLM)

Ollama + gemma3:4b를 사용한 무료 로컬 후처리 교정 기능입니다.

### 설치

```bash
# Ollama 설치 (macOS)
brew install ollama

# 모델 다운로드
ollama pull gemma3:4b

# Ollama 서버 실행
ollama serve
```

### 사용법

```python
from src.corrector import Corrector

corrector = Corrector()
corrected = corrector.correct("알루미늄 하품으로 바뀌고 있습니다")
# → "알루미늄 합금으로 바뀌고 있습니다"

# 도메인 힌트를 주면 전문 용어 교정 정확도 향상
corrected = corrector.correct(
    "알루미늄 하품으로 바뀌고 있습니다",
    domain="자동차 안전 기술 뉴스. 보닛, 알루미늄 합금, 충돌 흡수"
)
```

### 도메인 힌트

`correct(text, domain="...")` 파라미터로 도메인 맥락을 전달하면 교정 정확도가 올라갑니다.

- STT의 `initial_prompt`와는 별도 — `initial_prompt`에 구체적 단어를 넣으면 STT가 역효과
- 도메인 힌트는 LLM 교정에만 사용되므로 구체적 용어를 자유롭게 나열 가능

### 교정 예시 (EI_0001.mp4)

| STT 원본 | LLM 교정 | 변경 내용 |
|---------|---------|----------|
| 사고시 | 사고 시 | 띄어쓰기 |
| 흡수해줄 | 흡수해 줄 | 띄어쓰기 |
| 알루미늄 하품으로 | 알루미늄 합금으로 | 오인식 교정 |
| 안전평가에서 | 안전 평가에서 | 띄어쓰기 |

### 더 큰 모델 사용

교정 정확도가 부족하면 더 큰 모델을 사용할 수 있습니다.

```python
corrector = Corrector(model="gemma3:12b")  # RAM ~8GB 필요
```
