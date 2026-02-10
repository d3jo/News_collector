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
TIME_WINDOW_HOURS: int = 336

# 최대 기사 수 - Increased to accommodate more sources
MAX_ARTICLES: int = 300

# 카테고리 분류 - Improved with broader keywords
CATEGORIES = {
    "policy": [
        # Core policy terms
        "law", "regulation", "legislation", "bill", "act", "statute",
        "rule", "compliance", "legal", "regulatory",
        # Specific regulations
        "gdpr", "ccpa", "coppa", "pipeda", "lgpd", "dpa",
        # Actions
        "proposed", "enacted", "amended", "guidance", "framework",
        # Bodies
        "ftc", "ico", "edpb", "commission", "authority",
    ],
    
    "incident": [
        # Breach types  
        "breach", "hack", "hacked", "leak", "leaked", "exposed",
        "cyberattack", "attack", "incident", "vulnerability",
        "ransomware", "malware", "compromised",
        # Consequences
        "fine", "fined", "penalty", "settlement", "sued",
        "lawsuit", "litigation", "enforcement", "violation",
        "investigation", "investigated",
        # Reporting
        "notifies", "notification", "disclosed", "reports",
    ],
    
    "technology": [
        # AI
        "ai", "artificial intelligence", "machine learning",
        "algorithm", "automated", "ml", "chatbot", "gpt",
        # Biometrics
        "biometric", "facial recognition", "fingerprint",
        "face scan", "voice recognition",
        # Privacy tech
        "encryption", "anonymization", "tracking",
        "surveillance", "cookies", "location data",
        # Emerging
        "blockchain", "iot", "smart device", "wearable",
    ],
}