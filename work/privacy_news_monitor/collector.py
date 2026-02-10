"""뉴스 수집 모듈 - 다양한 소스에서 개인정보 보호 관련 뉴스 수집"""

import os
import re
import json
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from .config import SEARCH_KEYWORDS, TIME_WINDOW_HOURS, MAX_ARTICLES, CATEGORIES
import time
import socket




@dataclass
class Article:
    """뉴스 기사 데이터 클래스"""
    title: str
    url: str
    source: str
    published_date: datetime
    author: Optional[str] = None
    summary: Optional[str] = None
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "author": self.author,
            "summary": self.summary,
            "category": self.category,
        }


class NewsCollector:
    """뉴스 수집기 클래스"""

    def __init__(self, gnews_api_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        """
        Args:
            gnews_api_key: GNews API 키 (선택)
            newsapi_key: NewsAPI 키 (선택)
        """
        self.gnews_api_key = gnews_api_key or os.environ.get("GNEWS_API_KEY")
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY")
        
        # Additional news sources for more article coverage
        self.mediastack_key = os.environ.get("MEDIASTACK_API_KEY")
        self.currents_key = os.environ.get("CURRENTS_API_KEY")
        self.newsdata_key = os.environ.get("NEWSDATA_API_KEY")
        #self.bing_key = os.environ.get("BING_NEWS_API_KEY")
        
        self.articles: List[Article] = []

    def _sleep_between_requests(self, seconds: float = 0.4):
        time.sleep(seconds)




    def _make_request(self, url: str) -> Optional[Dict]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PrivacyNewsMonitor/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, socket.timeout) as e:
            print(f"요청 실패: {e} | URL: {url}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """다양한 날짜 형식 파싱"""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        return None
    
    
    def _deduplicate(self, articles: List[Article]) -> List[Article]:
        """Remove duplicate articles based on normalized title and URL"""
        seen = set()
        out = []
        for a in articles:
            # Normalize title: lowercase, remove extra whitespace
            title = " ".join((a.title or "").strip().lower().split())
            url = (a.url or "").strip().lower()
            
            key = (title, url)
            if key in seen or (not title and not url):
                continue
            seen.add(key)
            out.append(a)
        return out

    def _categorize_article(self, title: str, summary: str = "") -> str:
        """기사 카테고리 분류 (4개 카테고리만 허용: policy/incident/technology/general)"""
        allowed = {"policy", "incident", "technology", "general"}

        title_str = str(title) if title else ""
        summary_str = str(summary) if summary else ""
        text = (title_str + " " + summary_str).lower()

        # Only consider allowed categories from config
        for category, keywords in (CATEGORIES or {}).items():
            if category not in allowed or category == "general":
                continue
            if any(str(kw).lower() in text for kw in (keywords or [])):
                return category

        # fallback
        return "general"

    def _is_within_time_window(self, published_date: datetime) -> bool:
        """48시간 이내 기사인지 확인"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=TIME_WINDOW_HOURS)

        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)

        return published_date >= cutoff

    def collect_from_gnews(self, query: str) -> List[Article]:
        """GNews API에서 뉴스 수집"""
        if not self.gnews_api_key:
            return []

        articles = []
        encoded_query = urllib.parse.quote(query)
        # Increased from max=10 to max=20 to get more results
        url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&max=100&apikey={self.gnews_api_key}"

        data = self._make_request(url)
        if not data or "articles" not in data:
            return articles

        for item in data["articles"]:
            published_date = self._parse_date(item.get("publishedAt", ""))
            if not published_date:
                continue

            if not self._is_within_time_window(published_date):
                continue

            article = Article(
                title=item.get("title", ""),
                url=item.get("url", ""),
                source=item.get("source", {}).get("name", "Unknown"),
                published_date=published_date,
                author=item.get("author"),
                summary=item.get("description", ""),
                category=self._categorize_article(
                    item.get("title", ""),
                    item.get("description", "")
                ),
            )
            articles.append(article)

        return articles

    def collect_from_newsapi(self, query: str, max_pages: int = 3, page_size: int = 100) -> List[Article]:
        """NewsAPI에서 뉴스 수집 (pagination 지원: page)"""
        if not self.newsapi_key:
            return []

        articles: List[Article] = []
        from_date = (datetime.now(timezone.utc) - timedelta(hours=TIME_WINDOW_HOURS)).strftime("%Y-%m-%d")
        encoded_query = urllib.parse.quote(query)

        for page in range(1, max_pages + 1):
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={encoded_query}&from={from_date}&language=en"
                f"&pageSize={page_size}&page={page}&sortBy=publishedAt"
                f"&apiKey={self.newsapi_key}"
            )

            data = self._make_request(url)
            items = (data or {}).get("articles") or []
            if not items:
                break  # no more pages

            for item in items:
                published_date = self._parse_date(item.get("publishedAt", ""))
                if not published_date or not self._is_within_time_window(published_date):
                    continue

                articles.append(Article(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("name", "Unknown"),
                    published_date=published_date,
                    author=item.get("author"),
                    summary=item.get("description", ""),
                    category=self._categorize_article(item.get("title", ""), item.get("description", "")),
                ))

            # If fewer than page_size returned, likely last page
            if len(items) < page_size:
                break

            self._sleep_between_requests(0.4)

        return articles

    def collect_from_mediastack(self, query: str, max_pages: int = 4, page_size: int = 25) -> List[Article]:
        """MediaStack API에서 뉴스 수집 (pagination: offset)"""
        if not self.mediastack_key:
            return []

        articles: List[Article] = []
        encoded_query = urllib.parse.quote(query)

        for page in range(max_pages):
            offset = page * page_size
            url = (
                f"http://api.mediastack.com/v1/news?"
                f"access_key={self.mediastack_key}"
                f"&keywords={encoded_query}&languages=en"
                f"&limit={page_size}&offset={offset}&sort=published_desc"
            )

            data = self._make_request(url)
            items = (data or {}).get("data") or []
            if not items:
                break

            for item in items:
                published_date = self._parse_date(item.get("published_at", ""))
                if not published_date or not self._is_within_time_window(published_date):
                    continue

                articles.append(Article(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source=item.get("source", "MediaStack"),
                    published_date=published_date,
                    author=item.get("author"),
                    summary=item.get("description", ""),
                    category=self._categorize_article(item.get("title", ""), item.get("description", "")),
                ))

            if len(items) < page_size:
                break

            self._sleep_between_requests(0.5)

        return articles

    def collect_from_currents(self, query: str, max_pages: int = 5) -> List[Article]:
        """Currents API에서 뉴스 수집 (pagination: page_number)"""
        if not self.currents_key:
            return []

        articles: List[Article] = []
        encoded_query = urllib.parse.quote(query)

        for page in range(1, max_pages + 1):
            url = (
                f"https://api.currentsapi.services/v1/search?"
                f"keywords={encoded_query}&language=en"
                f"&page_number={page}"
                f"&apiKey={self.currents_key}"
            )

            data = self._make_request(url)
            items = (data or {}).get("news") or []
            if not items:
                break

            for item in items:
                published_date = self._parse_date(item.get("published", ""))
                if not published_date or not self._is_within_time_window(published_date):
                    continue

                articles.append(Article(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source=item.get("author", "Currents"),
                    published_date=published_date,
                    author=item.get("author"),
                    summary=item.get("description", ""),
                    category=self._categorize_article(item.get("title", ""), item.get("description", "")),
                ))

            self._sleep_between_requests(0.4)

        return articles

    def collect_from_newsdata(self, query: str, max_pages: int = 10) -> List[Article]:
        if not self.newsdata_key:
            return []

        articles: List[Article] = []
        encoded_query = urllib.parse.quote(query)

        next_page = None
        pages = 0

        while pages < max_pages:
            if next_page:
                url = (
                    f"https://newsdata.io/api/1/news?"
                    f"apikey={self.newsdata_key}&q={encoded_query}&language=en"
                    f"&page={urllib.parse.quote(next_page)}"
                )
            else:
                url = f"https://newsdata.io/api/1/news?apikey={self.newsdata_key}&q={encoded_query}&language=en"

            data = self._make_request(url)
            items = (data or {}).get("results") or []
            if not items:
                break

            for item in items:
                published_date = self._parse_date(item.get("pubDate", ""))
                if not published_date or not self._is_within_time_window(published_date):
                    continue

                creator = item.get("creator")
                author = creator[0] if isinstance(creator, list) and creator else None

                articles.append(Article(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    source=item.get("source_id", "NewsData"),
                    published_date=published_date,
                    author=author,
                    summary=item.get("description", ""),
                    category=self._categorize_article(item.get("title", ""), item.get("description", "")),
                ))

            next_page = (data or {}).get("nextPage")
            pages += 1
            if not next_page:
                break

            self._sleep_between_requests(0.5)

        return articles
      

    def collect_from_bing(self, query: str) -> List[Article]:
        """Bing News Search API에서 뉴스 수집 - 1000 transactions/month"""
        if not self.bing_key:
            return []

        articles = []
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.bing.microsoft.com/v7.0/news/search?q={encoded_query}&count=50&mkt=en-US"
        
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PrivacyNewsMonitor/1.0",
                    "Ocp-Apim-Subscription-Key": self.bing_key
                }
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            print(f"Bing API 요청 실패: {e}")
            return articles

        if not data or "value" not in data:
            return articles

        for item in data["value"]:
            published_date = self._parse_date(item.get("datePublished", ""))
            if not published_date:
                continue

            if not self._is_within_time_window(published_date):
                continue

            article = Article(
                title=item.get("name", ""),
                url=item.get("url", ""),
                source=item.get("provider", [{}])[0].get("name", "Bing News"),
                published_date=published_date,
                summary=item.get("description", ""),
                category=self._categorize_article(
                    item.get("name", ""),
                    item.get("description", "")
                ),
            )
            articles.append(article)

        return articles

    def collect_all(self) -> List[Article]:
        """모든 소스에서 뉴스 수집 - Now includes 6 sources instead of 2"""
        all_articles = []
        seen_urls = set()

        for keyword in SEARCH_KEYWORDS:
            # GNews에서 수집
            gnews_articles = self.collect_from_gnews(keyword)
            for article in gnews_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            # NewsAPI에서 수집
            newsapi_articles = self.collect_from_newsapi(keyword)
            for article in newsapi_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            # MediaStack에서 수집
            mediastack_articles = self.collect_from_mediastack(keyword)
            for article in mediastack_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            # Currents에서 수집
            currents_articles = self.collect_from_currents(keyword)
            for article in currents_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            # NewsData.io에서 수집
            newsdata_articles = self.collect_from_newsdata(keyword)
            for article in newsdata_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

            # Bing News에서 수집
            bing_articles = self.collect_from_bing(keyword)
            for article in bing_articles:
                if article.url not in seen_urls:
                    seen_urls.add(article.url)
                    all_articles.append(article)

        # 날짜순 정렬 (최신순)
        all_articles.sort(key=lambda x: x.published_date, reverse=True)

        # 최대 기사 수 제한
        self.articles = all_articles[:MAX_ARTICLES]
        return self.articles

    def add_sample_articles(self) -> List[Article]:
        """샘플 기사 추가 (API 키가 없을 때 사용)"""
        now = datetime.now(timezone.utc)

        sample_articles = [
            Article(
                title="New State Privacy Laws Expand Consumer Data Control in 2026",
                url="https://natlawreview.com/article/new-state-privacy-laws-expand-consumer-data-control-2026",
                source="National Law Review",
                published_date=now - timedelta(hours=6),
                author="Privacy Law Team",
                summary="Indiana, Kentucky, Rhode Island 등 3개 주의 새로운 개인정보보호법이 2026년 1월 1일부로 시행되어 미국 내 포괄적 개인정보보호법을 시행하는 주가 총 20개로 증가했습니다. 소비자들은 개인 데이터에 대한 접근, 수정, 삭제 및 타겟 광고, 판매, 프로파일링에 대한 거부권을 갖게 됩니다.",
                category="policy",
            ),
            Article(
                title="EDPB Selects Transparency as 2026 Coordinated Enforcement Focus",
                url="https://www.edpb.europa.eu/news/news/2025/coordinated-enforcement-framework-edpb-selects-topic-2026_en",
                source="European Data Protection Board",
                published_date=now - timedelta(hours=12),
                author="EDPB Press Office",
                summary="유럽데이터보호위원회(EDPB)가 2026년 공동 집행 주제로 'GDPR 제12-14조에 따른 투명성 및 정보 의무 준수'를 선정했습니다. 이는 개인정보 처리 시 정보주체에게 적절한 고지가 이루어지는지를 중점 점검할 예정입니다.",
                category="policy",
            ),
            Article(
                title="FTC Finalizes Order with General Motors Over Geolocation Data Collection",
                url="https://www.ftc.gov/news-events/news/press-releases",
                source="Federal Trade Commission",
                published_date=now - timedelta(hours=18),
                author="FTC Press Office",
                summary="FTC가 General Motors 및 OnStar와의 합의를 최종 확정했습니다. 이 합의는 소비자의 정밀 위치정보 및 운전 행동 데이터를 수집, 사용, 판매한 혐의에 관한 것으로, 향후 이러한 데이터 수집에 대한 명시적 동의를 요구합니다.",
                category="incident",
            ),
            Article(
                title="Blue Shield of California Notifies Members of Potential Data Breach",
                url="https://news.blueshieldca.com/january-5-2026-blue-shield-of-california-notifies-members-of-potential-data-breach",
                source="Blue Shield of California",
                published_date=now - timedelta(hours=24),
                author="Blue Shield Communications",
                summary="Blue Shield of California가 회원들의 보호건강정보(PHI)에 영향을 미칠 수 있는 잠재적 개인정보 침해에 대해 통지했습니다. 해당 사건에 대한 조사가 진행 중이며 영향받은 회원들에게 신용 모니터링 서비스를 제공할 예정입니다.",
                category="incident",
            ),
            Article(
                title="700Credit Data Breach Exposes 5.6 Million Records Including Social Security Numbers",
                url="https://www.brightdefense.com/news/700credit-breach/",
                source="Bright Defense",
                published_date=now - timedelta(hours=20),
                author="Security Research Team",
                summary="자동차 딜러십용 신용보고서 제공업체 700Credit LLC가 약 560만 명의 개인정보 유출을 공개했습니다. 유출된 정보에는 이름, 주소, 생년월일, 사회보장번호가 포함되어 있으며 사우스캐롤라이나주에서만 108,829명이 영향을 받았습니다.",
                category="incident",
            ),
            Article(
                title="EU-UK Adequacy Decision Renewed Until 2031",
                url="https://www.gibsondunn.com/gibson-dunn-europe-data-protection-january-2026/",
                source="Gibson Dunn",
                published_date=now - timedelta(hours=8),
                author="Privacy Practice Group",
                summary="유럽위원회가 GDPR 및 법집행지침에 따른 영국에 대한 적정성 결정을 2031년 12월 27일까지 연장했습니다. 이로써 EU에서 영국으로의 개인정보 이전이 별도의 추가 조치 없이 계속 가능해졌습니다.",
                category="policy",
            ),
            Article(
                title="California CCPA Regulations for Automated Decision-Making Now Effective",
                url="https://iapp.org/news/a/new-year-new-rules-us-state-privacy-requirements-coming-online-as-2026-begins",
                source="IAPP",
                published_date=now - timedelta(hours=15),
                author="IAPP Staff",
                summary="캘리포니아 소비자 프라이버시법(CCPA)의 자동화된 의사결정 기술, 위험 평가 및 사이버보안 감사에 관한 규정이 2026년 1월 1일부터 적용됩니다. 또한 민감 개인정보의 정의가 신경 데이터와 16세 미만 미성년자 데이터를 포함하도록 확대되었습니다.",
                category="technology",
            ),
            Article(
                title="ICO Announces New AI and Automated Decision-Making Guidance for 2026",
                url="https://ico.org.uk/about-the-ico/media-centre/news-and-blogs/2026/01/ai-ll-get-that/",
                source="UK ICO",
                published_date=now - timedelta(hours=10),
                author="ICO Communications",
                summary="영국 정보위원회(ICO)가 2026년 중 자동화된 의사결정 및 프로파일링에 대한 가이드라인 업데이트, AI 및 자동화된 의사결정에 대한 법정 행동강령, 에이전틱 AI의 개인정보보호 영향에 대한 호라이즌 스캐닝 보고서를 발표할 예정입니다.",
                category="technology",
            ),
            Article(
                title="DeepSeek AI App Raises Privacy and Security Concerns",
                url="https://krebsonsecurity.com/2025/02/experts-flag-security-privacy-risks-in-deepseek-ai-app/",
                source="Krebs on Security",
                published_date=now - timedelta(hours=22),
                author="Brian Krebs",
                summary="중국 AI 기업 DeepSeek의 앱이 하드코딩된 암호화 키 사용, 중국 기업으로의 암호화되지 않은 사용자 데이터 전송 등 심각한 보안 및 개인정보 위험을 노출하고 있습니다. 미 해군, 텍사스주, 대만, 이탈리아가 사용을 금지했습니다.",
                category="technology",
            ),
            Article(
                title="Colorado's Algorithmic Accountability Law Takes Effect February 2026",
                url="https://www.hunton.com/privacy-and-information-security-law/new-u-s-state-privacy-social-media-and-ai-laws-take-effect-in-january-2026",
                source="Hunton Andrews Kurth",
                published_date=now - timedelta(hours=14),
                author="Privacy & Data Security Practice",
                summary="콜로라도주의 알고리즘 책임법이 2026년 2월부터 시행됩니다. 고용, 의료, 교육 결정에 사용되는 고위험 AI 시스템을 정의하고, 개발자에게 문서화 및 차별 완화 의무를 부과하며, 소비자에게 고지, 설명, 수정, 이의제기권을 부여합니다.",
                category="policy",
            ),
            Article(
                title="EU AI Act Full Implementation in August 2026 to Prohibit Eight Unacceptable Practices",
                url="https://www.pearlcohen.com/new-privacy-data-protection-and-ai-laws-in-2026/",
                source="Pearl Cohen",
                published_date=now - timedelta(hours=16),
                author="IP & Tech Team",
                summary="EU AI법이 2026년 8월 전면 시행되어 유해한 조작 및 무차별 안면인식 스크래핑 등 8가지 금지 관행을 규정합니다. 채용, 법집행, 핵심 인프라의 고위험 AI 시스템은 적절한 위험 평가, 활동 로그 유지, 인간 감독을 입증해야 합니다.",
                category="policy",
            ),
            Article(
                title="FTC Issues $10 Million Settlement with Disney for COPPA Violations",
                url="https://www.ftc.gov/news-events/news/press-releases",
                source="Federal Trade Commission",
                published_date=now - timedelta(hours=30),
                author="FTC Press Office",
                summary="연방법원이 Disney의 아동 온라인 프라이버시 보호법(COPPA) 위반 혐의에 대해 1,000만 달러의 합의를 승인했습니다. 이번 조치는 아동 개인정보 보호에 대한 FTC의 강력한 집행 의지를 보여줍니다.",
                category="incident",
            ),
            Article(
                title="New York AI Transparency Laws Require Disclosure of Synthetic Performers",
                url="https://www.privacyworld.blog/2026/01/primer-on-2026-consumer-privacy-ai-and-cybersecurity-laws/",
                source="Privacy World",
                published_date=now - timedelta(hours=26),
                author="Privacy World Staff",
                summary="뉴욕주 상원 법안 S8420A에 따라 광고주는 AI나 소프트웨어 알고리즘으로 생성된 '합성 퍼포머'를 사용할 때 이를 명확히 공개해야 합니다. 최초 위반 시 1,000달러, 재위반 시 5,000달러의 민사 벌금이 부과됩니다.",
                category="policy",
            ),
            Article(
                title="US Lawmakers Introduce Bill to Restrict ICE's Mobile Facial Recognition",
                url="https://www.biometricupdate.com/202601/lawmakers-move-to-rein-in-ices-use-of-mobile-facial-recognition",
                source="Biometric Update",
                published_date=now - timedelta(hours=28),
                author="Biometric Update Staff",
                summary="미 하원 의원 Bennie G. Thompson이 국토안보부(DHS)의 출입국 이외 지역에서의 모바일 생체인식 앱 사용을 금지하고 미국 시민에게서 수집된 생체정보 파기를 요구하는 법안을 발의했습니다.",
                category="policy",
            ),
            Article(
                title="COPPA Rule Amendments Expand Biometric Data Protection for Children",
                url="https://www.dwt.com/blogs/privacy--security-law-blog/2025/05/coppa-rule-ftc-amended-childrens-privacy",
                source="Davis Wright Tremaine",
                published_date=now - timedelta(hours=32),
                author="Privacy Team",
                summary="FTC의 COPPA 규칙 개정으로 보호 대상 개인정보가 지문, 홍채 패턴, 성문, 안면 인식 템플릿 등 생체인식 정보를 포함하도록 확대되었습니다. 기업은 2026년 4월 22일까지 규정을 준수해야 합니다.",
                category="policy",
            ),
            Article(
                title="UK ICO Launches Consultation on New Data Protection Enforcement Guidance",
                url="https://ico.org.uk/about-the-ico/ico-and-stakeholder-consultations/2025/10/ico-consultation-on-data-protection-enforcement-procedural-guidance/",
                source="UK ICO",
                published_date=now - timedelta(hours=34),
                author="ICO Legal Team",
                summary="영국 ICO가 데이터보호 집행 절차 지침 초안에 대한 공개 의견수렴을 시작했습니다. 이 지침은 2025년 데이터(사용 및 접근)법에 의해 도입된 인터뷰 참석 요구, 전문가 보고서 위탁 등 새로운 조사 권한의 사용 방법을 제시합니다.",
                category="policy",
            ),
            Article(
                title="California Delete Act DROP Platform Launches with New Data Broker Requirements",
                url="https://cppa.ca.gov/announcements/",
                source="California Privacy Protection Agency",
                published_date=now - timedelta(hours=36),
                author="CPPA Staff",
                summary="캘리포니아 삭제법의 삭제 요청 및 거부 플랫폼(DROP)이 출시되어 새로운 데이터 브로커 요건과 연간 브로커 등록 관련 기존 요건 이상의 벌금 규정이 도입되었습니다. 데이터 브로커는 1월 31일까지 등록을 완료해야 합니다.",
                category="policy",
            ),
            Article(
                title="Consortium of Privacy Regulators Strengthens Multi-State Enforcement Coordination",
                url="https://www.mwe.com/insights/data-privacy-and-cybersecurity-developments-we-are-watching-in-2026/",
                source="McDermott Will & Emery",
                published_date=now - timedelta(hours=38),
                author="Privacy & Cybersecurity Team",
                summary="2025년에 설립된 개인정보보호 규제 기관 컨소시엄이 전문성과 자원을 공유하고 법 위반 가능성에 대한 조사 노력을 조율하고 있습니다. 캘리포니아, 콜로라도, 코네티컷 규제 당국이 공동 조사에 협력하고 있습니다.",
                category="incident",
            ),
            Article(
                title="Civil Service Employees Association Data Breach Affects 47,000 People",
                url="https://www.claimdepot.com/investigations/civil-service-employees-association-data-breach-2026",
                source="Claim Depot",
                published_date=now - timedelta(hours=40),
                author="Investigation Team",
                summary="CSEA(공무원노동조합)가 2026년 1월부터 영향받은 개인들에게 서면 통지를 시작했습니다. 이번 침해 사고는 전국적으로 47,352명에게 영향을 미쳤으며 민감한 개인정보가 노출된 것으로 알려졌습니다.",
                category="incident",
            ),
            Article(
                title="EU Digital Omnibus Package Proposes GDPR Amendments Amid Privacy Concerns",
                url="https://techgdpr.com/blog/data-protection-digest-03012026-improvements-are-being-made-to-gdpr-enforcement-us-consumer-privacy-and-emerging-shadow-ai/",
                source="TechGDPR",
                published_date=now - timedelta(hours=42),
                author="TechGDPR Editorial",
                summary="EU가 GDPR 및 관련 규정에 대한 표적화된 개정을 제안하는 디지털 옴니버스 패키지를 도입했습니다. 규정 준수 간소화, AI 혁신 촉진, 침해 신고 절차 개선을 목표로 하나, 프라이버시 옹호자들은 보호 약화를 우려하고 있습니다.",
                category="policy",
            ),
        ]

        self.articles = sample_articles
        return self.articles


def create_collector(gnews_api_key: Optional[str] = None, newsapi_key: Optional[str] = None) -> NewsCollector:
    """NewsCollector 인스턴스 생성 팩토리 함수"""
    return NewsCollector(gnews_api_key=gnews_api_key, newsapi_key=newsapi_key)