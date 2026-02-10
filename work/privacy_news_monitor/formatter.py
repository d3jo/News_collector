"""포맷터 모듈 - 기사 요약 및 마크다운 출력 생성"""

from datetime import datetime, timezone
from typing import List, Dict, Optional
from .collector import Article


class ArticleFormatter:
    """기사 포맷터 클래스"""

    CATEGORY_LABELS = {
        "policy": "🏛️ 정책/규제",
        "incident": "⚠️ 사건/조사",
        "technology": "🔬 신기술",
        "general": "📰 일반",
    }

    CATEGORY_LABELS_NO_EMOJI = {
        "policy": "[정책/규제]",
        "incident": "[사건/조사]",
        "technology": "[신기술]",
        "general": "[일반]",
    }

    def __init__(self, use_emoji: bool = False):
        """
        Args:
            use_emoji: 이모지 사용 여부
        """
        self.use_emoji = use_emoji

    def _get_category_label(self, category: str) -> str:
        """카테고리 라벨 반환"""
        if self.use_emoji:
            return self.CATEGORY_LABELS.get(category, self.CATEGORY_LABELS["general"])
        return self.CATEGORY_LABELS_NO_EMOJI.get(category, self.CATEGORY_LABELS_NO_EMOJI["general"])

    def _format_date(self, dt: datetime) -> str:
        """날짜 포맷팅"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M UTC")

    def _format_author(self, author: Optional[str]) -> str:
        """저자 정보 포맷팅"""
        if author:
            return f"작성자: {author}"
        return "작성자: 미상"

    def format_single_article(self, article: Article, index: int) -> str:
        """단일 기사 마크다운 포맷팅"""
        category_label = self._get_category_label(article.category)
        date_str = self._format_date(article.published_date)
        author_str = self._format_author(article.author)

        lines = [
            f"### {index}. {article.title} [[링크]]({article.url})",
            f"",
            f"**{category_label}** | {article.source} | {date_str} | {author_str}",
            f"",
        ]

        if article.summary:
            lines.append(f"> {article.summary}")
        else:
            lines.append(f"> (요약 없음)")

        lines.append("")
        return "\n".join(lines)

    def format_article_list(self, articles: List[Article]) -> str:
        """전체 기사 목록 마크다운 포맷팅"""
        if not articles:
            return "## 지난 48시간 내 수집된 기사가 없습니다.\n"

        now = datetime.now(timezone.utc)
        header = [
            "# 개인정보 보호 동향 모니터링 리포트",
            "",
            f"**생성일시:** {now.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**기준 기간:** 지난 48시간",
            f"**수집 기사 수:** {len(articles)}건",
            "",
            "---",
            "",
        ]

        # 카테고리별 통계
        category_counts = {}
        for article in articles:
            cat = article.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        stats_lines = ["## 카테고리별 분포", ""]
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            label = self._get_category_label(cat)
            stats_lines.append(f"- {label}: {count}건")
        stats_lines.extend(["", "---", ""])

        # 기사 목록
        article_lines = ["## 기사 목록", ""]
        for i, article in enumerate(articles, 1):
            article_lines.append(self.format_single_article(article, i))

        return "\n".join(header + stats_lines + article_lines)

    def format_by_category(self, articles: List[Article]) -> str:
        """카테고리별로 그룹화된 마크다운 포맷팅"""
        if not articles:
            return "## 지난 48시간 내 수집된 기사가 없습니다.\n"

        now = datetime.now(timezone.utc)
        header = [
            "# 개인정보 보호 동향 모니터링 리포트",
            "",
            f"**생성일시:** {now.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**기준 기간:** 지난 48시간",
            f"**수집 기사 수:** {len(articles)}건",
            "",
            "---",
            "",
        ]

        # 카테고리별 그룹화
        categorized: Dict[str, List[Article]] = {}
        for article in articles:
            cat = article.category
            if cat not in categorized:
                categorized[cat] = []
            categorized[cat].append(article)

        # 카테고리 순서 정의
        category_order = ["policy", "incident", "technology", "general"]

        content_lines = []
        article_num = 1

        for cat in category_order:
            if cat not in categorized:
                continue

            cat_articles = categorized[cat]
            label = self._get_category_label(cat)

            content_lines.append(f"## {label} ({len(cat_articles)}건)")
            content_lines.append("")

            for article in cat_articles:
                content_lines.append(self.format_single_article(article, article_num))
                article_num += 1

            content_lines.append("---")
            content_lines.append("")

        return "\n".join(header + content_lines)

    def format_simple_list(self, articles: List[Article]) -> str:
        """간단한 목록 형태의 마크다운 포맷팅"""
        if not articles:
            return "지난 48시간 내 수집된 기사가 없습니다.\n"

        lines = [
            "# 개인정보 보호 동향 - 최근 48시간",
            "",
        ]

        for i, article in enumerate(articles, 1):
            date_str = self._format_date(article.published_date)
            author_info = f" ({article.author})" if article.author else ""
            lines.append(f"{i}. **{article.title}** [[링크]]({article.url})")
            lines.append(f"   - {article.source} | {date_str}{author_info}")
            if article.summary:
                lines.append(f"   - {article.summary}")
            lines.append("")

        return "\n".join(lines)


def create_formatter(use_emoji: bool = False) -> ArticleFormatter:
    """ArticleFormatter 인스턴스 생성 팩토리 함수"""
    return ArticleFormatter(use_emoji=use_emoji)
