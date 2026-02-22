from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str, language: str = None, initial_prompt: str = None) -> list[dict]:
        segments, info = self.model.transcribe(
            audio_path,
            # language: 언어 코드 (예: "ko", "en", "ja"). None이면 자동 감지
            # 언어를 알면 무조건 지정 / 잘못된 언어를 넣는 것 보다 None이 더 안전 ex) Hello world가 헬로 워르드의 결과가 나올 수 있음
            language=language,
            # word_timestamps: True면 단어별 시작/끝 시간 제공
            word_timestamps=True,
            # initial_prompt: 모델에 힌트를 주는 텍스트. 도메인/맥락 정보를 넣으면 인식률 향상
            # 모델이 모든 구간에서 단어들을 우선적으로 찾으려 함 / 짧고 일반적인 맥락만 주는 것이 좋음 / 구체적인 단어를 나열하면 역효과가 남
            initial_prompt=initial_prompt,
            # ── VAD (Voice Activity Detection) 설정 ──
            # vad_filter: True면 무음 구간을 자동으로 건너뜀
            vad_filter=True,
            vad_parameters=dict(
                # threshold: 음성 감지 민감도 (0~1). 낮을수록 작은 소리도 음성으로 인식
                # 기본 값 : 0.5
                # 0.1로 낮추면 배경 소음까지 음성으로 잡음
                threshold=0.3,
                # min_silence_duration_ms: 이 시간(ms) 이상 무음이면 문장 끊김으로 판단
                # 기본 값 :: 2000
                # 사람이 말할 때 숨 쉬는 간격이 보통 0.2~0.5
                min_silence_duration_ms=300,
                # speech_pad_ms: 감지된 음성 앞뒤에 추가하는 여유 시간(ms). 잘림 방지
                # 기본 값 : 30
                # 특히 인터뷰처럼 갑자기 시작하는 음성의 첫 음절이 보존됨
                speech_pad_ms=200,
            ),
            # ── 기타 유용한 옵션 (현재 기본값 사용 중) ──
            #
            # beam_size=5
            #   음성을 듣고 단어를 고를 때, 동시에 몇 개의 후보를 유지하면서 탐색할지 정하는 값
            #   각 단계에서 상위 N개 후보를 유지하며 탐색
            #   기본값: 5 / 높이면(10): 더 많은 후보를 탐색해서 정확도↑, 속도 2배↓
            #   1로 설정하면 greedy 디코딩 (가장 빠르지만 정확도 낮음)
            #   주의: 너무 높이면(10+) 오히려 이상한 후보를 선택할 수 있음
            #
            # temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            #   모델이 단어를 선택할 때 얼마나 모험적으로 고를지 정하는 값
            #   한번에 다 쓰는 게 아니라 단계적 폴백
            #   0.0: 가장 확신 높은 단어만 선택 (결정적)
            #   1.0: 다양한 단어를 고려 (창의적, 오류 가능성↑)
            #   보수적 설정: [0.0, 0.2, 0.4] / 기본값이 대부분 적절 / 0.6 이상의 "모험적 선택"을 아예 차단
            #
            # log_prob_threshold=-1.0
            #   모델이 인식 결과를 얼마나 확신하는지 판단하는 기준
            #   세그먼트의 평균 로그 확률이 이 값보다 낮으면 폴백/스킵
            #   기본값: -1.0 / 낮추면(-1.5): 확신이 낮은 인식도 허용 (누락 방지)
            #   높이면(-0.5): 확신 높은 인식만 남김 (오인식 제거, 하지만 누락 증가)
            #
            # compression_ratio_threshold=2.4
            # 인식 결과가 반복 오류인지 판단하는 기준 / 텍스트를 gzip으로 압축해서 압축률을 봄
            # 텍스트가 반복적일수록 압축률이 높아짐 (예: "네 네 네 네" → 압축 잘됨)
            # 이 값을 넘으면 "반복 오류"로 판단하고 temperature를 올려 재시도
            # 기본값: 2.4 / 낮추면(2.0): 반복 감지 엄격 / 높이면(2.8): 반복 허용 관대
            #
            # repetition_penalty=1.0
            #   이미 나온 단어를 다시 선택할 때 확률을 깎는 값
            #   기본값: 1.0 (확률 그대로) / 1.2~1.5: 반복 단어의 확률을 낮춤
            #   주의: 너무 높이면 실제로 반복되는 정상 발화("보행자의 충돌 충격")도 다른 단어로 바뀔 수 있음
            #
            # no_repeat_ngram_size=0
            #   N-gram 반복 방지. 설정한 N개 단어 조합이 반복되면 차단
            #   기본값: 0 (비활성화) / 2: 같은 2단어 조합 반복 차단
            #   예: "네 네 네 네" 같은 반복 제거에 유용
            #
            # condition_on_previous_text=True
            #   이전 세그먼트의 인식 결과를 다음 세그먼트의 컨텍스트로 사용
            #   True: 문맥 연결이 자연스러움 (기본값)
            #   False: 각 세그먼트를 독립적으로 인식. 반복 루프 오류가 생기면 False로 변경
            #
            # no_speech_threshold=0.6
            #   무음 판단 임계값 (0~1). "이 구간이 음성이 아닐 확률"
            #   기본값: 0.6 / 높이면(0.8): 배경음 필터 강화 (조용한 음성도 잘릴 수 있음)
            #   낮추면(0.3): 더 관대하게 음성으로 판단 (노이즈도 포함될 수 있음)
            #
            # hotwords=None
            #   자주 등장하는 단어 힌트. initial_prompt와 비슷하지만 매 세그먼트에 적용
            #   예: "합금, 보닛, 와이퍼"
            #   주의: initial_prompt와 같이 사용 가능하지만, prefix와는 같이 사용 불가
            #   전문 용어 교정은 결국 Claude 후처리
        )
        results = []
        for segment in segments:
            words = []
            if segment.words:
                for w in segment.words:
                    words.append({
                        "start": w.start,
                        "end": w.end,
                        "word": w.word,
                    })
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": words,
            })
        return results

    @staticmethod
    def split_into_sentences(results: list[dict]) -> list[dict]:
        """세그먼트를 문장 단위로 분리한다. word_timestamps 기반으로
        마침표(.), 물음표(?), 느낌표(!) 위치에서 문장을 끊는다."""
        sentences = []
        for r in results:
            if not r["words"]:
                sentences.append({
                    "start": r["start"],
                    "end": r["end"],
                    "text": r["text"].strip(),
                })
                continue

            current_words = []
            sentence_start = None

            for w in r["words"]:
                if sentence_start is None:
                    sentence_start = w["start"]
                current_words.append(w["word"])

                if w["word"].rstrip().endswith((".", "?", "!")):
                    sentences.append({
                        "start": sentence_start,
                        "end": w["end"],
                        "text": "".join(current_words).strip(),
                    })
                    current_words = []
                    sentence_start = None

            # 문장부호 없이 끝난 나머지
            if current_words:
                sentences.append({
                    "start": sentence_start,
                    "end": r["words"][-1]["end"],
                    "text": "".join(current_words).strip(),
                })

        return sentences

    def transcribe_text(self, audio_path: str, language: str = None) -> str:
        results = self.transcribe(audio_path, language=language)
        return " ".join(r["text"].strip() for r in results)
