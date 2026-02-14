"""
Enhanced news summarization and deduplication module with Sentence-BERT
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from openai import OpenAI

# Sentence-BERT for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")


@dataclass
class ArticleSummary:
    """뉴스 기사 요약 데이터 클래스"""
    title: str
    source: str
    bullets: List[str]
    embedding: Optional[np.ndarray] = None
    original_desc: Optional[str] = None


class SBertDeduplicator:
    """Sentence-BERT 기반 중복 제거 클래스"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        """
        Args:
            model_name: Sentence-BERT 모델 이름
                - paraphrase-multilingual-mpnet-base-v2: 다국어 지원 (한국어 우수), 높은 품질
                - paraphrase-multilingual-MiniLM-L12-v2: 빠른 속도, 괜찮은 품질
        """
        self.model = None
        self.model_name = model_name
        
        if SBERT_AVAILABLE:
            try:
                print(f"Sentence-BERT 모델 로딩 중: {model_name}")
                self.model = SentenceTransformer(model_name)
                print("✓ Sentence-BERT 모델 로드 완료")
            except Exception as e:
                print(f"Sentence-BERT 모델 로드 실패: {e}")
    
    def create_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트의 임베딩 벡터 생성"""
        if not self.model:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"임베딩 생성 실패: {e}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def deduplicate(
        self, 
        summaries: List[ArticleSummary], 
        similarity_threshold: float = 0.85
    ) -> Tuple[List[ArticleSummary], List[Dict]]:
        """
        Sentence-BERT 임베딩 기반 중복 제거
        
        Args:
            summaries: 중복 제거할 요약 리스트
            similarity_threshold: 유사도 임계값
                - 0.90+ : 매우 유사한 기사만 중복으로 간주 (보수적)
                - 0.85-0.87 : 균형잡힌 설정 (권장) ✓
                - 0.80- : 다소 다른 기사도 중복으로 간주 (공격적)
        
        Returns:
            (중복 제거된 요약 리스트, 제거된 항목 정보)
        """
        if not self.model or not summaries:
            return summaries, []
        
        print(f"\n{'='*60}")
        print(f"Sentence-BERT 임베딩 기반 중복 제거 시작")
        print(f"{'='*60}")
        print(f"입력 기사 수: {len(summaries)}")
        print(f"유사도 임계값: {similarity_threshold}")
        
        # 1. 임베딩 생성
        print("\n임베딩 벡터 생성 중...")
        for idx, summary in enumerate(summaries):
            if idx % 10 == 0:
                print(f"  진행: {idx}/{len(summaries)}")
            
            if summary.embedding is None:
                # 불릿포인트를 하나의 텍스트로 결합
                text = " ".join(summary.bullets) if summary.bullets else summary.title
                summary.embedding = self.create_embedding(text)
        
        # 2. 유사도 기반 중복 제거
        print("\n유사도 계산 및 중복 제거 중...")
        keep_indices = []
        removed_items = []
        
        for i in range(len(summaries)):
            if i % 20 == 0:
                print(f"  진행: {i}/{len(summaries)}")
            
            is_duplicate = False
            
            # 이미 유지하기로 결정한 기사들과 비교
            for j in keep_indices:
                if summaries[i].embedding is None or summaries[j].embedding is None:
                    continue
                
                similarity = self.cosine_similarity(
                    summaries[i].embedding,
                    summaries[j].embedding
                )
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    removed_items.append({
                        "kept_index": j,
                        "removed_index": i,
                        "similarity": similarity,
                        "kept_title": summaries[j].title,
                        "removed_title": summaries[i].title,
                        "kept_bullets": summaries[j].bullets,
                        "removed_bullets": summaries[i].bullets
                    })
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        # 3. 결과 출력
        deduplicated = [summaries[i] for i in keep_indices]
        
        print(f"\n{'='*60}")
        print(f"Sentence-BERT 중복 제거 완료")
        print(f"{'='*60}")
        print(f"입력: {len(summaries)}개")
        print(f"출력: {len(deduplicated)}개")
        print(f"제거: {len(removed_items)}개")
        
        if removed_items:
            print(f"\n제거된 중복 기사 샘플 (최대 5개):")
            for item in removed_items[:5]:
                print(f"\n  유사도: {item['similarity']:.3f}")
                print(f"  유지 [{item['kept_index']}]: {item['kept_title'][:80]}")
                print(f"  제거 [{item['removed_index']}]: {item['removed_title'][:80]}")
        
        return deduplicated, removed_items


# Singleton instances
_openai_client = None
_sbert_deduplicator = None


def get_client():
    """OpenAI 클라이언트 싱글톤"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def get_sbert_deduplicator(model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
    """SBert Deduplicator 싱글톤"""
    global _sbert_deduplicator
    if _sbert_deduplicator is None:
        _sbert_deduplicator = SBertDeduplicator(model_name=model_name)
    return _sbert_deduplicator


def summarize_korean_bullets(title: str, desc: str | None, source: str) -> List[str]:
    """
    Summarize news article in Korean with 3-5 bullet points.
    
    Args:
        title: Article title
        desc: Article description/content
        source: Article source
        
    Returns:
        List of Korean summary bullet points
    """
    client = get_client()

    seed = desc or ""
    prompt = f"""다음 뉴스 기사를 한국어로 3~5개의 불릿포인트로 요약해 주세요.
각 불릿포인트는 핵심 내용만 간결하게 작성하세요.

제목: {title}
출처: {source}
내용:
{seed}

요약 (불릿포인트로):"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes news in Korean."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        text = response.choices[0].message.content.strip()
        
        # Parse bullet points
        bullets = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
        
        # Filter out empty lines and ensure we have content
        bullets = [b for b in bullets if b and len(b) > 5]
        
        return bullets if bullets else ["요약을 생성할 수 없습니다."]
        
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise


def summarize_and_deduplicate(
    articles: List[Dict[str, str]],
    similarity_threshold: float = 0.85,
    sbert_model: str = "paraphrase-multilingual-mpnet-base-v2"
) -> Tuple[List[ArticleSummary], List[Dict]]:
    """
    뉴스 기사를 요약하고 Sentence-BERT로 중복 제거
    
    Args:
        articles: 기사 리스트, 각 항목은 {'title': str, 'desc': str, 'source': str} 형태
        similarity_threshold: 유사도 임계값 (0.85-0.87 권장)
        sbert_model: Sentence-BERT 모델 이름
        
    Returns:
        (중복 제거된 요약 리스트, 제거된 항목 정보)
        
    Example:
        >>> articles = [
        ...     {"title": "뉴스1", "desc": "내용1", "source": "출처1"},
        ...     {"title": "뉴스2", "desc": "내용2", "source": "출처2"},
        ... ]
        >>> summaries, removed = summarize_and_deduplicate(articles, similarity_threshold=0.85)
        >>> for summary in summaries:
        ...     print(f"{summary.title}: {summary.bullets}")
    """
    if not articles:
        return [], []
    
    print(f"\n{'='*60}")
    print(f"뉴스 요약 및 중복 제거 프로세스 시작")
    print(f"{'='*60}")
    print(f"총 기사 수: {len(articles)}")
    
    # 1. 한국어 요약 생성
    print(f"\n1단계: 한국어 불릿포인트 요약 생성 중...")
    summaries: List[ArticleSummary] = []
    
    for idx, article in enumerate(articles):
        if idx % 5 == 0:
            print(f"  진행: {idx}/{len(articles)}")
        
        title = article.get("title", "")
        desc = article.get("desc") or article.get("summary") or article.get("description")
        source = article.get("source", "Unknown")
        
        try:
            bullets = summarize_korean_bullets(title, desc, source)
            summaries.append(ArticleSummary(
                title=title,
                source=source,
                bullets=bullets,
                original_desc=desc
            ))
        except Exception as e:
            print(f"  경고: 기사 요약 실패 (인덱스 {idx}): {e}")
            # 실패해도 제목만이라도 포함
            summaries.append(ArticleSummary(
                title=title,
                source=source,
                bullets=[title],
                original_desc=desc
            ))
    
    print(f"1단계 완료: {len(summaries)}개 요약 생성됨")
    
    # 2. Sentence-BERT 중복 제거
    print(f"\n2단계: Sentence-BERT 중복 제거")
    deduplicator = get_sbert_deduplicator(model_name=sbert_model)
    deduplicated, removed = deduplicator.deduplicate(summaries, similarity_threshold)
    
    print(f"\n{'='*60}")
    print(f"프로세스 완료")
    print(f"{'='*60}")
    print(f"최종 결과: {len(deduplicated)}개 고유 기사")
    
    return deduplicated, removed


def batch_summarize_and_deduplicate(
    articles: List[Dict[str, str]],
    batch_size: int = 50,
    similarity_threshold: float = 0.85,
    sbert_model: str = "paraphrase-multilingual-mpnet-base-v2"
) -> Tuple[List[ArticleSummary], List[Dict]]:
    """
    대량의 뉴스 기사를 배치로 나눠서 요약하고 중복 제거
    
    Args:
        articles: 기사 리스트
        batch_size: 배치 크기 (OpenAI rate limit 고려)
        similarity_threshold: 유사도 임계값
        sbert_model: Sentence-BERT 모델 이름
        
    Returns:
        (중복 제거된 요약 리스트, 제거된 항목 정보)
    """
    if not articles:
        return [], []
    
    print(f"\n{'='*60}")
    print(f"대량 뉴스 배치 처리 시작")
    print(f"{'='*60}")
    print(f"총 기사 수: {len(articles)}")
    print(f"배치 크기: {batch_size}")
    
    all_summaries = []
    
    # 배치 단위로 요약 생성
    for i in range(0, len(articles), batch_size):
        batch = articles[i:i + batch_size]
        print(f"\n배치 {i//batch_size + 1}/{(len(articles)-1)//batch_size + 1} 처리 중...")
        
        summaries, _ = summarize_and_deduplicate(
            batch,
            similarity_threshold=1.0,  # 배치 내에서는 중복 제거 안함
            sbert_model=sbert_model
        )
        all_summaries.extend(summaries)
        
        # Rate limiting
        if i + batch_size < len(articles):
            import time
            time.sleep(2)
    
    # 전체 중복 제거
    print(f"\n전체 배치에 대한 중복 제거 수행 중...")
    deduplicator = get_sbert_deduplicator(model_name=sbert_model)
    final_summaries, removed = deduplicator.deduplicate(all_summaries, similarity_threshold)
    
    return final_summaries, removed


# Example usage
if __name__ == "__main__":
    # 테스트 데이터
    test_articles = [
        {
            "title": "EU 최고 법원이 WhatsApp의 EU 개인정보 보호 감시 기간에 대한 싸움을 하급 법원으로 되돌림",
            "desc": "유럽 최고 법원이 WhatsApp의 EU 개인정보 보호 감시 기간과의 법적 분쟁을 하급 법원으로 되돌림. 이 사건은 아일랜드 데이터 보호 당국이 WhatsApp에 2억 2500만 유로(약 2억 6800만 달러)의 벌금을 부과하라는 명령에서 시작됨.",
            "source": "The Star"
        },
        {
            "title": "WhatsApp의 EU 개인정보 보호 전투, 법원이 사건을 다시 승화하며 재점화",
            "desc": "유럽 최고 법원이 WhatsApp과 EU 간의 개인정보 보호 당국 간의 분쟁을 재점화하며 사건을 하급 법원으로 송환. 이 사건은 아일랜드 데이터 보호 당국이 WhatsApp에 부과한 2억 2천5백만 유로의 벌금과 관련됨.",
            "source": "Devdiscourse"
        },
        {
            "title": "대법원, WhatsApp-Meta 개인정보 보호 정책 사건을 2월 23일로 연기하다",
            "desc": "대법원이 WhatsApp과 Meta의 개인정보 보호 정책 사건을 2026년 2월 23일로 연기함. 사용자 데이터 보호에 대한 우려가 사건 연기의 배경으로 작용함.",
            "source": "The Hindu"
        },
    ]
    
    # 요약 및 중복 제거 실행
    summaries, removed_items = summarize_and_deduplicate(
        test_articles,
        similarity_threshold=0.85  # 권장 임계값
    )
    
    print(f"\n{'='*60}")
    print(f"최종 결과")
    print(f"{'='*60}")
    
    for idx, summary in enumerate(summaries, 1):
        print(f"\n[{idx}] {summary.title}")
        print(f"출처: {summary.source}")
        print(f"요약:")
        for bullet in summary.bullets:
            print(f"  • {bullet}")