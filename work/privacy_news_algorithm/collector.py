"""뉴스 수집 모듈 - 다양한 소스에서 개인정보 보호 관련 뉴스 수집 (중복 제거 개선)"""

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
from difflib import SequenceMatcher
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# Semantic deduplication via sentence embeddings
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print(
        "Warning: sentence-transformers not installed. "
        "Semantic deduplication will be skipped. "
        "Run: pip install sentence-transformers"
    )


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


def normalize_url(url: str) -> str:
    """URL 정규화 - tracking 파라미터 제거 + host/path 정리"""
    if not url:
        return ""

    url = url.strip()

    try:
        p = urlparse(url)
    except Exception:
        return url.lower().split("#")[0].split("?")[0].rstrip("/").strip()

    scheme = "https"
    netloc = (p.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]

    path = (p.path or "").strip()
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    drop_prefixes = ("utm_",)
    drop_exact = {
        "gclid", "fbclid", "yclid", "mc_cid", "mc_eid", "ref", "ref_src",
        "cmpid", "cmid", "rss", "rssid"
    }

    qs = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = (k or "").lower().strip()
        if any(kl.startswith(pref) for pref in drop_prefixes):
            continue
        if kl in drop_exact:
            continue
        qs.append((kl, (v or "").strip()))

    qs.sort()
    query = urlencode(qs, doseq=True)

    return urlunparse((scheme, netloc, path, "", query, ""))


def normalize_title(title: str) -> str:
    """제목 정규화 - 소스 접미사/특수문자/공백 정리"""
    if not title:
        return ""

    t = title.lower().strip()
    t = re.sub(r"\s*\|\s*.+$", "", t)
    t = re.sub(r"\s*-\s*reuters\s*$", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = " ".join(t.split())
    return t


def titles_are_similar(title1: str, title2: str, threshold: float = 0.85) -> bool:
    """두 제목이 유사한지 확인 (85% 유사도 기준)"""
    if not title1 or not title2:
        return False

    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    if norm1 == norm2:
        return True

    ratio = SequenceMatcher(None, norm1, norm2).ratio()
    return ratio >= threshold


class NewsCollector:
    """뉴스 수집기 클래스"""

    def __init__(self, gnews_api_key: Optional[str] = None, newsapi_key: Optional[str] = None):
        self.gnews_api_key = gnews_api_key or os.environ.get("GNEWS_API_KEY")
        self.newsapi_key = newsapi_key or os.environ.get("NEWSAPI_KEY")

        self.mediastack_key = os.environ.get("MEDIASTACK_API_KEY")
        self.currents_key = os.environ.get("CURRENTS_API_KEY")
        self.newsdata_key = os.environ.get("NEWSDATA_API_KEY")
        self.rapidapi_key = os.environ.get("RAPIDAPI_KEY")

        self.articles: List[Article] = []

        # Load semantic embedding model once at startup.
        # all-MiniLM-L6-v2 is ~90MB, fast, and well-suited for short texts like news titles.
        # Downloaded automatically on first run and cached by sentence-transformers.
        if SBERT_AVAILABLE:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Warning: Could not load sentence-transformer model: {e}")
                self._embedder = None
        else:
            self._embedder = None

    def _sleep_between_requests(self, seconds: float = 1.2):
        time.sleep(seconds)

    def _make_request_with_headers(self, url: str, headers: Dict[str, str]) -> Optional[Dict]:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"Rate limit (429) hit for: {url}. Sleeping 5 seconds...")
                time.sleep(5)
            else:
                print(f"HTTP Error {e.code}: {e} | URL: {url}")
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, socket.timeout) as e:
            print(f"요청 실패: {e} | URL: {url}")
            return None

    def _make_request(self, url: str) -> Optional[Dict]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PrivacyNewsMonitor/1.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"Rate limit (429) hit for: {url}. Sleeping 5 seconds...")
                time.sleep(5)
            else:
                print(f"HTTP Error {e.code}: {e} | URL: {url}")
            return None
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, socket.timeout) as e:
            print(f"요청 실패: {e} | URL: {url}")
            return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
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

    def _is_duplicate(self, article: Article, seen_urls: set, seen_titles: List[str]) -> bool:
        norm_url = normalize_url(article.url)
        norm_title = normalize_title(article.title)

        if norm_url and norm_url in seen_urls:
            return True

        for seen_title in seen_titles:
            if titles_are_similar(norm_title, seen_title):
                return True

        return False

    def _deduplicate(self, articles: List[Article]) -> List[Article]:
        """
        Three-stage deduplication:
        1) Canonical URL exact match
        2) Normalized title exact match + fuzzy SequenceMatcher (within bucket)
        3) TF-IDF cosine similarity on title+summary (catches same-story, different-wording)
        """
        if not articles:
            return []

        # Stage 1 & 2: URL + title dedup (existing logic)
        articles = self._deduplicate_structural(articles)

        # Stage 3: Semantic dedup via sentence-transformers
        if SBERT_AVAILABLE and self._embedder is not None and len(articles) > 1:
            articles = self._deduplicate_by_tfidf(articles)

        return articles

    def _deduplicate_structural(self, articles: List[Article]) -> List[Article]:
        """
        Original dedup logic: URL exact match + title exact/fuzzy match.
        Extracted into its own method so _deduplicate stays clean.
        """
        if not articles:
            return []

        ordered = sorted(
            articles,
            key=lambda x: x.published_date or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )

        by_url: Dict[str, Article] = {}
        by_title: Dict[str, Article] = {}
        buckets: Dict[str, List[str]] = {}

        def better(a: Article, b: Article) -> Article:
            da = a.published_date or datetime.min.replace(tzinfo=timezone.utc)
            db = b.published_date or datetime.min.replace(tzinfo=timezone.utc)
            if db > da:
                a, b = b, a
            sa = len((a.summary or "").strip())
            sb = len((b.summary or "").strip())
            if sb > sa:
                a = b
            return a

        out: List[Article] = []

        for art in ordered:
            norm_url = normalize_url(art.url)
            norm_title = normalize_title(art.title)

            if not norm_url and not norm_title:
                continue

            # 1) URL dedupe
            if norm_url:
                existing = by_url.get(norm_url)
                if existing:
                    by_url[norm_url] = better(existing, art)
                    continue

            # 2) exact title dedupe
            if norm_title:
                existing = by_title.get(norm_title)
                if existing:
                    by_title[norm_title] = better(existing, art)
                    continue

            # 3) fuzzy title dedupe (bucketed by first 8 words)
            is_dup = False
            if norm_title:
                words = norm_title.split()
                bucket_key = " ".join(words[:8]) if words else norm_title
                candidates = buckets.get(bucket_key, [])

                for seen_norm_title in candidates:
                    if SequenceMatcher(None, norm_title, seen_norm_title).ratio() >= 0.92:
                        is_dup = True
                        break

                if is_dup:
                    continue

                buckets.setdefault(bucket_key, []).append(norm_title)

            out.append(art)
            if norm_url:
                by_url[norm_url] = art
            if norm_title:
                by_title[norm_title] = art

        return out

    def _deduplicate_by_tfidf(
        self,
        articles: List[Article],
        threshold: float = 0.85,
        same_day_only: bool = True,
    ) -> List[Article]:
        """
        Semantic deduplication using sentence-transformer embeddings.

        Unlike TF-IDF (which only measures word overlap), sentence-transformers
        map text into a semantic vector space — so "Delhi HC notice" and
        "Delhi High Court Challenges" are understood as nearly identical,
        even though they share almost no words.

        Args:
            articles:      Pre-structurally-deduped article list.
            threshold:     Cosine similarity cutoff (0.0–1.0).
                           0.85 is a good default for news deduplication:
                           - Raise to 0.90 if unrelated articles are being incorrectly merged.
                           - Lower to 0.80 if same-story duplicates are still slipping through.
            same_day_only: Only compare articles published on the same UTC date.
                           Prevents false positives on recurring topics (e.g. annual
                           "Privacy Day" articles) and speeds up the O(n²) pass.

        Returns:
            Deduplicated list. When two articles are duplicates, the one with
            the longer summary is kept. Order is preserved (newest-first).
        """
        if not SBERT_AVAILABLE or self._embedder is None or len(articles) < 2:
            return articles

        # Title repeated twice to weight it more heavily than the summary,
        # since titles are the most reliable signal for same-story detection.
        texts = [
            f"{a.title} {a.title} {a.summary or ''}".strip()
            for a in articles
        ]

        try:
            # Encode all articles in one batched call (fast even for 100s of articles)
            embeddings = self._embedder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                batch_size=64,
            )
        except Exception as e:
            print(f"Warning: Semantic embedding failed, skipping semantic dedup: {e}")
            return articles

        n = len(articles)
        kept = [True] * n

        for i in range(n):
            if not kept[i]:
                continue

            for j in range(i + 1, n):
                if not kept[j]:
                    continue

                # Only compare articles from the same calendar day to
                # avoid false positives on similar-but-unrelated recurring topics
                if same_day_only:
                    date_i = (articles[i].published_date or datetime.min.replace(tzinfo=timezone.utc)).date()
                    date_j = (articles[j].published_date or datetime.min.replace(tzinfo=timezone.utc)).date()
                    if date_i != date_j:
                        continue

                sim = float(st_util.cos_sim(embeddings[i], embeddings[j]))

                if sim >= threshold:
                    # Keep whichever has the longer summary; discard the other
                    len_i = len((articles[i].summary or "").strip())
                    len_j = len((articles[j].summary or "").strip())
                    if len_j > len_i:
                        kept[i] = False
                        break  # i is gone; stop comparing it
                    else:
                        kept[j] = False

        return [a for a, keep in zip(articles, kept) if keep]

    def _categorize_article(self, title: str, summary: str = "") -> str:
        """
        개선된 카테고리 분류 - 구체적인 카테고리 우선

        Priority order (most specific → most general):
        1. incident (breaches, hacks, investigations, fines)
        2. technology (AI, encryption, new tech, tools)
        3. policy (laws, regulations, compliance)
        4. general (fallback)
        """
        title_str = str(title) if title else ""
        summary_str = str(summary) if summary else ""
        text = (title_str + " " + summary_str).lower()

        incident_keywords = [
            "breach", "breached", "hack", "hacked", "cyberattack", "cyber-attack",
            "ransomware", "malware", "data leak", "leaked", "exposed", "compromised",
            "investigation", "investigated", "probe", "probing",
            "lawsuit", "sued", "suing", "settlement",
            "fine", "fined", "penalty", "penalties",
            "charged", "indicted", "convicted",
            "violation", "violated", "violating",
            "ftc action", "enforcement action", "crackdown",
            "scam", "fraud", "phishing", "identity theft",
            "unauthorized access", "security incident"
        ]

        technology_keywords = [
            "artificial intelligence", "machine learning", "ai model", "ai system",
            "large language model", "llm", "chatgpt", "generative ai",
            "facial recognition", "face recognition", "biometric", "fingerprint",
            "encryption", "encrypted", "end-to-end encryption",
            "anonymization", "pseudonymization",
            "privacy-enhancing technology", "pet",
            "zero-knowledge", "differential privacy",
            "cookie", "third-party cookie", "tracking pixel",
            "browser fingerprinting", "device fingerprint",
            "vpn", "privacy tool", "privacy app",
            "data minimization tool", "consent management",
            "blockchain", "cryptocurrency", "web3",
            "tiktok privacy", "meta privacy", "google privacy settings"
        ]

        policy_keywords = [
            "gdpr", "general data protection regulation",
            "ccpa", "california consumer privacy act",
            "cpra", "california privacy rights act",
            "coppa", "hipaa", "ferpa", "glba",
            "bill", "legislation", "legislative",
            "privacy law", "privacy act", "data protection act",
            "privacy legislation", "privacy bill",
            "regulation", "regulatory framework",
            "compliance requirement", "regulatory compliance",
            "privacy framework", "policy framework",
            "ftc", "federal trade commission",
            "fcc", "sec", "attorney general",
            "privacy commission", "data protection authority",
            "dpa", "supervisory authority",
            "court ruling", "legal opinion", "guidance",
            "congressional", "senate", "parliament"
        ]

        for keyword in incident_keywords:
            if keyword in text:
                return "incident"

        for keyword in technology_keywords:
            if keyword in text:
                return "technology"

        for keyword in policy_keywords:
            if keyword in text:
                return "policy"

        return "general"

    def _is_within_time_window(self, published_date: datetime) -> bool:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=TIME_WINDOW_HOURS)
        if published_date.tzinfo is None:
            published_date = published_date.replace(tzinfo=timezone.utc)
        return published_date >= cutoff

    def collect_from_gnews(self, query: str) -> List[Article]:
        if not self.gnews_api_key:
            return []

        articles = []
        encoded_query = urllib.parse.quote(query)
        url = f"https://gnews.io/api/v4/search?q={encoded_query}&lang=en&max=100&apikey={self.gnews_api_key}"

        data = self._make_request(url)
        if not data or "articles" not in data:
            return articles

        for item in data["articles"]:
            published_date = self._parse_date(item.get("publishedAt", ""))
            if not published_date or not self._is_within_time_window(published_date):
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
                break

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

            if len(items) < page_size:
                break

            self._sleep_between_requests(1.0)

        return articles

    def collect_from_mediastack(self, query: str, max_pages: int = 4, page_size: int = 25) -> List[Article]:
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

            self._sleep_between_requests(1.2)

        return articles

    def collect_from_currents(self, query: str, max_pages: int = 50) -> List[Article]:
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
            if not data:
                break

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

            self._sleep_between_requests(1.2)

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
            if not data:
                break

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
                    url=item.get("url", item.get("link", "")),
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

            self._sleep_between_requests(1.2)

        return articles

    def collect_from_rapidapi(
        self,
        query: str,
        host: str,
        endpoint: str,
        max_pages: int = 5,
        page_size: int = 50,
        query_param: str = "q",
        page_param: Optional[str] = "page",
        page_size_param: str = "limit",
        items_path: str = "articles",
        title_field: str = "title",
        url_field: str = "url",
        date_field: str = "published",
        summary_field: str = "description",
        source_field: str = "source",
        author_field: Optional[str] = "author",
        extra_params: Optional[Dict[str, str]] = None,
        source_name: str = "RapidAPI",
    ) -> List[Article]:
        if not self.rapidapi_key:
            return []

        articles: List[Article] = []
        encoded_query = urllib.parse.quote(query)

        headers = {
            "X-RapidAPI-Key": self.rapidapi_key,
            "X-RapidAPI-Host": host,
            "User-Agent": "PrivacyNewsMonitor/1.0"
        }

        for page in range(1, max_pages + 1):
            params = {query_param: encoded_query, page_size_param: str(page_size)}

            if page_param:
                params[page_param] = str(page)

            if extra_params:
                params.update(extra_params)

            query_string = urllib.parse.urlencode(params)

            if not endpoint.startswith("http"):
                url = f"https://{host}{endpoint}?{query_string}"
            else:
                url = f"{endpoint}?{query_string}"

            data = self._make_request_with_headers(url, headers)

            if not data:
                break

            items = data
            for key in items_path.split("."):
                items = items.get(key, []) if isinstance(items, dict) else []
                if not items:
                    break

            if not items or not isinstance(items, list):
                break

            for item in items:
                date_str = item.get(date_field, "")
                published_date = self._parse_date(date_str) if date_str else None

                if not published_date or not self._is_within_time_window(published_date):
                    continue

                source = source_name
                if "." in source_field:
                    parts = source_field.split(".")
                    source_data = item
                    for part in parts:
                        source_data = source_data.get(part, {}) if isinstance(source_data, dict) else {}
                    source = source_data if isinstance(source_data, str) else source_name
                else:
                    source = item.get(source_field, source_name)

                author = None
                if author_field:
                    author = item.get(author_field)

                articles.append(Article(
                    title=item.get(title_field, ""),
                    url=item.get(url_field, ""),
                    source=source if source else source_name,
                    published_date=published_date,
                    author=author,
                    summary=item.get(summary_field, ""),
                    category=self._categorize_article(
                        item.get(title_field, ""),
                        item.get(summary_field, "")
                    ),
                ))

            if len(items) < page_size:
                break

            self._sleep_between_requests(1.2)

        return articles

    def collect_all(self) -> List[Article]:
        """모든 소스에서 뉴스 수집 - 개선된 중복 제거 적용"""
        all_articles = []

        for keyword in SEARCH_KEYWORDS:
            all_articles.extend(self.collect_from_gnews(keyword))
            all_articles.extend(self.collect_from_newsapi(keyword))
            all_articles.extend(self.collect_from_mediastack(keyword))
            all_articles.extend(self.collect_from_currents(keyword))
            all_articles.extend(self.collect_from_newsdata(keyword))

            try:
                rapidapi_articles = self.collect_from_rapidapi(
                    query=keyword,
                    host="real-time-news-data.p.rapidapi.com",
                    endpoint="https://real-time-news-data.p.rapidapi.com/search",
                    max_pages=5,
                    page_size=50,
                    query_param="query",
                    page_param="page",
                    page_size_param="limit",
                    items_path="data",
                    title_field="title",
                    url_field="link",
                    date_field="published_datetime_utc",
                    summary_field="snippet",
                    source_field="source_name",
                    extra_params={"lang": "en"},
                    source_name="RapidAPI-RealTimeNews",
                )
                all_articles.extend(rapidapi_articles)
            except Exception as e:
                print(f"RapidAPI 수집 중 오류: {e}")

        # Stage 1+2: URL / title dedup  →  Stage 3: TF-IDF semantic dedup
        all_articles = self._deduplicate(all_articles)

        all_articles.sort(key=lambda x: x.published_date, reverse=True)
        self.articles = all_articles[:MAX_ARTICLES]
        return self.articles


def create_collector(gnews_api_key: Optional[str] = None, newsapi_key: Optional[str] = None) -> NewsCollector:
    """NewsCollector 인스턴스 생성 팩토리 함수"""
    return NewsCollector(gnews_api_key=gnews_api_key, newsapi_key=newsapi_key)