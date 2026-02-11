import os
from openai import OpenAI

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)

def summarize_korean_bullets(title: str, desc: str | None, source: str):
    """
    Summarize news article in Korean with 3-5 bullet points.
    
    Args:
        title: Article title
        desc: Article description/content
        source: Article source
        
    Returns:
        List of Korean summary bullet points
    """
    client = get_client()  # Create client lazily, AFTER dotenv loads

    seed = desc or ""
    prompt = f"""다음 뉴스 기사를 한국어로 3~5개의 불릿포인트로 요약해 주세요.
각 불릿포인트는 핵심 내용만 간결하게 작성하세요.

제목: {title}
출처: {source}
내용:
{seed}

요약 (불릿포인트로):"""

    try:
        response = client.chat.completions.create(  # Correct method
            model="gpt-4o-mini",
            messages=[  # Correct parameter
                {"role": "system", "content": "You are a helpful assistant that summarizes news in Korean."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        text = response.choices[0].message.content.strip()  # Correct extraction
        
        # Parse bullet points
        bullets = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        
        # Filter out empty lines and ensure we have content
        bullets = [b for b in bullets if b and len(b) > 5]
        
        return bullets if bullets else ["요약을 생성할 수 없습니다."]
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise  # Re-raise to be caught by caller