import os
from src.transcriber import Transcriber


def test_transcribe():
    audio_path = os.path.join("audio", "test.wav")
    if not os.path.exists(audio_path):
        print(f"테스트 오디오 파일이 없습니다: {audio_path}")
        return

    transcriber = Transcriber(model_size="base")
    results = transcriber.transcribe(audio_path)

    for r in results:
        print(f"[{r['start']:.2f}s -> {r['end']:.2f}s] {r['text']}")
    assert len(results) > 0


if __name__ == "__main__":
    test_transcribe()
