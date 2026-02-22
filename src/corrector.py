import json
import urllib.request

SYSTEM_PROMPT = (
    "당신은 한국어 음성인식(STT) 결과를 교정하는 전문가입니다.\n"
    "규칙:\n"
    "1. 문맥상 잘못 인식된 단어만 교정하세요\n"
    "2. 원래 문장의 의미와 구조를 유지하세요\n"
    "3. 맞춤법과 띄어쓰기를 교정하세요\n"
    "4. 교정된 텍스트만 출력하세요. 설명은 하지 마세요"
)


class Corrector:
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def correct(self, text: str, domain: str = "") -> str:
        domain_hint = f"\n도메인: {domain}" if domain else ""
        payload = json.dumps({
            "model": self.model,
            "prompt": f"다음 STT 결과를 교정해주세요:{domain_hint}\n\n{text}",
            "system": SYSTEM_PROMPT,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["response"]
