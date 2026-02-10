"""설정 파일 - 개인정보 보호 뉴스 모니터링 서비스"""

from typing import List

# 검색 키워드 (개인정보 보호 관련) - Very specific to avoid irrelevant results
SEARCH_KEYWORDS = [
    "privacy law",
    "data protection",
    "GDPR",
    "CCPA",
    "COPPA",
    "data breach",
    "privacy investigation",
    "privacy regulation",
    "AI privacy",
    "personal data",
]

# 신뢰할 수 있는 뉴스 소스
TRUSTED_SOURCES: List[str] = [
    "iapp.org",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "theregister.com",
    "zdnet.com",
    "reuters.com",
    "bloomberg.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "bbc.com",
    "politico.com",
    "lawfaremedia.org",
    "eff.org",
    "epic.org",
    "hunton.com",
    "natlawreview.com",
    "gibsondunn.com",
    "osborneclarke.com",
]

# 시간 설정 (시간 단위) - Increased to 7 days to get more articles
TIME_WINDOW_HOURS: int = 168

# 최대 기사 수 - Increased to accommodate more sources
MAX_ARTICLES: int = 50

# 카테고리 분류
CATEGORIES = {
    "policy": ["privacy policy", "privacy regulation", "privacy law", "privacy legislation", "privacy act", "privacy bill", "gdpr", "ccpa", "coppa", "privacy compliance", "privacy statute"],
    "incident": ["privacy breach", "privacy hack", "privacy leak", "privacy investigation", "privacy fine", "privacy penalty", "privacy enforcement", "privacy violation", "data exposure"],
    "technology": ["privacy ai", "privacy artificial intelligence", "privacy machine learning", "privacy biometric", "privacy facial recognition", "privacy encryption", "privacy tool", "privacy technology"],
}