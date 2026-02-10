import os
from openai import OpenAI

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)

def summarize_korean_bullets(title: str, desc: str | None, source: str):
    client = get_client()  # ← create client lazily, AFTER dotenv loads

    seed = desc or ""
    prompt = f"""
다음 뉴스 기사를 한국어로 3~5개의 불릿포인트로 무엇에 대한 것인지, 어떤 내용인지 요약해 주세요.

제목: {title}
출처: {source}
내용:
{seed}
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    text = resp.output_text.strip()
    bullets = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    return bullets
