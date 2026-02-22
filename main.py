import os
import sys
from src.transcriber import Transcriber
from src.corrector import Corrector


def load_file(path: str, default: str = "") -> str:
    """파일에서 텍스트를 읽는다. 파일이 없으면 기본값 반환."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return default


def main():
    if len(sys.argv) < 2:
        print("사용법: python main.py <오디오파일> [언어코드] [프롬프트파일] [도메인파일]")
        print("예시:  python main.py audio/EI_0001.mp4 ko audio/EI_0001.txt audio/EI_0001.domain.txt")
        sys.exit(1)

    audio_path = sys.argv[1]
    language = sys.argv[2] if len(sys.argv) > 2 else None
    prompt = load_file(sys.argv[3], "한국어 음성입니다.") if len(sys.argv) > 3 else "한국어 음성입니다."
    domain = load_file(sys.argv[4]) if len(sys.argv) > 4 else ""

    print(f"프롬프트: {prompt}")
    if domain:
        print(f"도메인: {domain}")
    print()

    # ──────────────────────────────────────────────────────────────
    # [기본 모델] (다국어 지원, .en 붙으면 영어 전용)
    #   tiny    (39M,  ~150MB) - 가장 빠름, 정확도 낮음. 리소스 제한 환경용
    #   base    (74M,  ~300MB) - 빠른 속도, 낮은 정확도. 간단한 테스트용
    #   small   (244M, ~1GB)   - 속도/정확도 균형. 일반 용도 추천
    #   medium  (769M, ~3GB)   - 높은 정확도. 프로덕션 품질 필요 시
    #   large-v3(1.5B, ~6GB)   - 최고 정확도. 전문 전사/다국어에 최적
    #
    # [경량화 모델] (distilled - 원본 대비 50% 작고 6배 빠름, 정확도 1% 이내 손실)
    #   distil-small.en        - small 경량화, 영어 전용
    #   distil-medium.en       - medium 경량화, 영어 전용
    #   distil-large-v2        - large-v2 경량화, 다국어
    #   distil-large-v3        - large-v3 경량화, 다국어. 가성비 최고
    #
    # [터보 모델] (디코더 32층→4층 축소, large-v3 대비 6배 빠름)
    #   large-v3-turbo (809M, ~3GB) - large-v3급 정확도 + 빠른 속도. 실시간용 추천
    # ──────────────────────────────────────────────────────────────
    transcriber = Transcriber(model_size="large-v3-turbo")
    results = transcriber.transcribe(
        audio_path,
        language=language,
        initial_prompt=prompt,
    )

    # 문장 단위로 분리 (마침표/물음표/느낌표 기준)
    sentences = transcriber.split_into_sentences(results)

    print("=" * 60)
    print("[ STT 원본 결과 ]")
    print("=" * 60)
    for s in sentences:
        print(f"[{s['start']:.1f}s -> {s['end']:.1f}s] {s['text']}")

    # 후처리 교정 (Ollama 로컬 LLM)
    raw_text = " ".join(s["text"] for s in sentences)
    corrector = Corrector()
    print("\n교정 중...")
    corrected = corrector.correct(raw_text, domain=domain)

    print()
    print("=" * 60)
    print("[ LLM 교정 결과 ]")
    print("=" * 60)
    for line in corrected.split(". "):
        line = line.strip()
        if line:
            print(line if line.endswith((".","?","!")) else line + ".")


if __name__ == "__main__":
    main()
