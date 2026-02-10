import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from textwrap import wrap


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))



from privacy_news_algorithm.collector import create_collector, Article
from privacy_news_algorithm.formatter import create_formatter
from privacy_news_algorithm.summarizer import summarize_korean_bullets

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(page_title="Privacy News Monitor", page_icon="🛡️", layout="wide")

# Custom CSS to make download button more prominent
st.markdown("""
<style>
    /* Make download button bigger and more colorful */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 16px 32px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stDownloadButton > button:active {
        transform: translateY(0px) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 실시간 개인정보 보안 동향 모니터링 리포트")
st.caption("Refresh to fetch the latest privacy-related news and categorize it.")

# --- Sidebar: API key status ---
st.sidebar.header("API Keys")
gnews_ok = bool(os.getenv("GNEWS_API_KEY"))
newsapi_ok = bool(os.getenv("NEWSAPI_KEY"))
openai_ok = bool(os.getenv("OPENAI_API_KEY"))
mediastack_ok = bool(os.getenv("MEDIASTACK_API_KEY"))
currents_ok = bool(os.getenv("CURRENTS_API_KEY"))
newsdata_ok = bool(os.getenv("NEWSDATA_API_KEY"))
rapidapi_ok = bool(os.getenv("RAPIDAPI_KEY"))

st.sidebar.write(f"GNEWS_API_KEY: {'✅' if gnews_ok else '❌'}")
st.sidebar.write(f"NEWSAPI_KEY: {'✅' if newsapi_ok else '❌'}")
st.sidebar.write(f"OPENAI_API_KEY: {'✅' if openai_ok else '❌'}")
st.sidebar.write(f"MEDIASTACK_API_KEY: {'✅' if mediastack_ok else '❌'}")
st.sidebar.write(f"CURRENTS_API_KEY: {'✅' if currents_ok else '❌'}")
st.sidebar.write(f"NEWSDATA_API_KEY: {'✅' if newsdata_ok else '❌'}")
st.sidebar.write(f"RAPIDAPI_KEY: {'✅' if rapidapi_ok else '❌'}")


st.sidebar.divider()
#use_emoji = st.sidebar.checkbox("Use emoji", value=True)
format_choice = st.sidebar.selectbox("Output format", ["category", "simple", "detailed"], index=0)



# --- Main controls ---
st.subheader("Latest Privacy Monitor")
st.caption("One click. Latest privacy news. Categorized into 정책/규제, 사건/조사, 신기술, 기타.")

run = st.button("🔄 Refresh", type="primary")


MONITOR_QUERY = ( '"data privacy" OR "privacy law" OR "privacy regulation" OR ' '"privacy breach" OR GDPR OR privacy OR ' '"privacy policy" OR "privacy violation" OR "privacy protection" OR ' '"data protection" OR "personal data" OR "consumer privacy" OR ' '"privacy rights" OR "user privacy" OR "information privacy"' )
# Privacy-focused search query - cast a wider net, let relevance filter refine
QUERY_NEWSAPI = MONITOR_QUERY
QUERY_SIMPLE = "data privacy OR privacy law OR GDPR OR privacy breach OR data protection OR consumer privacy"



def markdown_to_pdf(md_text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER

    # Margins
    x_margin = 1 * inch
    y_margin = 1 * inch
    y = height - y_margin

    # USE KOREAN-SAFE FONT
    c.setFont("HYSMyeongJo-Medium", 10)

    for line in md_text.split("\n"):
        # Wrap long lines safely
        wrapped_lines = wrap(line, 90, replace_whitespace=False)
        if not wrapped_lines:
            y -= 14

        for wl in wrapped_lines:
            if y < y_margin:
                c.showPage()
                c.setFont("HYSMyeongJo-Medium", 10)
                y = height - y_margin

            c.drawString(x_margin, y, wl)
            y -= 14

    c.save()
    buffer.seek(0)
    return buffer.read()


def dedupe_keep_latest(articles: list[Article]) -> list[Article]:
    """Remove duplicate articles based on URL and similar titles"""
    seen_urls = set()
    seen_titles = set()
    out = []
    
    for a in sorted(articles, key=lambda x: x.published_date or "", reverse=True):
        url = (a.url or "").strip().lower()
        # Normalize title for comparison (lowercase, remove extra spaces)
        title = " ".join((a.title or "").lower().split())
        
        # Skip if empty or already seen
        if not url and not title:
            continue
            
        if url and url in seen_urls:
            continue
            
        if title and title in seen_titles:
            continue
        
        # Add to seen sets
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
            
        out.append(a)
    
    return out

def is_privacy_relevant(article: Article) -> bool:
    """
    Score-based privacy relevance filter.
    - Strong privacy terms count more
    - Adjacent terms count less
    - Uses title weighting (title hits matter more)
    """
    title = (article.title or "").lower()
    body = ((article.summary or "") + " " + (getattr(article, "description", "") or "")).lower()
    text = f"{title} {body}"

    strong = [
        "privacy", "gdpr", "ccpa", "cpra", "coppa", "hipaa", "ferpa",
        "data protection", "personal data", "personal information",
        "data subject", "consent", "opt out", "right to delete",
        "data minimization", "purpose limitation", "surveillance",
        "biometric", "facial recognition", "geolocation", "location data",
        "data broker", "cookie", "tracking", "adtech"
    ]

    adjacent = [
        "breach", "leak", "hack", "ransomware", "security incident",
        "regulator", "enforcement", "fine", "lawsuit", "investigation",
        "compliance", "audit", "risk assessment", "ai act", "algorithm",
        "identity theft", "credential", "exposed records", "phishing"
    ]

    # Scoring
    score = 0

    # Title hits matter more
    score += sum(3 for t in strong if t in title)
    score += sum(1 for t in strong if t in body)

    score += sum(2 for t in adjacent if t in title)
    score += sum(1 for t in adjacent if t in body)

    # Threshold: lower = looser, higher = stricter
    return score >= 2



def pick_20_diverse(articles: list[Article], per_cat: int = 5, total: int = 20) -> list[Article]:
    """Pick diverse articles across categories, avoiding duplicates"""
    ordered = sorted(articles, key=lambda x: x.published_date or "", reverse=True)

    buckets = {"policy": [], "incident": [], "technology": [], "general": []}
    seen_urls = set()
    seen_titles = set()

    for a in ordered:
        # Skip duplicates
        url = (a.url or "").strip().lower()
        title = " ".join((a.title or "").lower().split())
        
        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue
            
        cat = a.category if a.category in buckets else "general"
        if len(buckets[cat]) < per_cat:
            buckets[cat].append(a)
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)

    out = []
    for cat in ["policy", "incident", "technology", "general"]:
        out.extend(buckets[cat])

    # Fill remaining slots with newest leftovers
    for a in ordered:
        if len(out) >= total:
            break
            
        url = (a.url or "").strip().lower()
        title = " ".join((a.title or "").lower().split())
        
        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue
            
        out.append(a)
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

    return out[:total]





@st.cache_data(show_spinner=False)
def translate_title_ko(title: str) -> str:
    if not title:
        return title
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=f"""Translate this news headline into natural Korean.
Rules:
- Keep proper nouns (company/product/person names) in English if commonly used.
- Keep it short like a real Korean headline.
- Output ONLY the translated headline.

Headline: {title}"""
    )
    return resp.output_text.strip()



# Cache summaries so reruns don't re-bill you
@st.cache_data(show_spinner=False)
def cached_korean_md(title, desc, source):
    bullets = summarize_korean_bullets(title, desc, source)
    return "\n".join([f"- {b}" for b in bullets])

if run:
    collector = create_collector(
        gnews_api_key=os.getenv("GNEWS_API_KEY"),
        newsapi_key=os.getenv("NEWSAPI_KEY"),
    )

    
    with st.spinner("Collecting latest articles..."):
        articles = []
        articles.extend(collector.collect_from_gnews(QUERY_SIMPLE))  # Already max at 100
        articles.extend(collector.collect_from_newsapi(QUERY_NEWSAPI, max_pages=5))  # 5→500 articles
        articles.extend(collector.collect_from_mediastack(QUERY_SIMPLE, max_pages=8))  # 8→200 articles
        articles.extend(collector.collect_from_currents(QUERY_SIMPLE, max_pages=10))  # 10 pages
        articles.extend(collector.collect_from_newsdata(QUERY_SIMPLE, max_pages=15))  # 15 pages
        articles.extend(
            collector.collect_from_rapidapi(
                QUERY_SIMPLE,
                host="real-time-news-data.p.rapidapi.com",
                endpoint="https://real-time-news-data.p.rapidapi.com/search",
                max_pages=3,  # Increased
                page_size=10,  # Increased
                query_param="query",
                page_param="page",  # Changed from "time_published"
                page_size_param="limit",
                items_path="data",
                title_field="title",
                url_field="link",
                date_field="published_datetime_utc",
                summary_field="snippet",
                source_field="source_name",
                extra_params={"lang": "en"},  # Removed "country": "US" to get global news
                source_name="RapidAPI-RealTimeNews",
            )
        )
        
        # Deduplicate first
        articles = dedupe_keep_latest(articles)
        
        # Filter for privacy relevance (optional)
        privacy_articles = [a for a in articles if is_privacy_relevant(a)]
        st.info(f"Found {len(articles)} articles, {len(privacy_articles)} are privacy-relevant (strict filter enabled)")
        articles = privacy_articles
        st.info(f"Found {len(articles)} articles (strict filter disabled)")
        
        # Pick diverse set from articles
        articles = pick_20_diverse(articles, per_cat=5, total=20)

    if not articles:
        st.warning("No privacy-relevant articles found. This could mean:")
        st.info("• Not enough recent privacy news in the past 48 hours\n• API returned mostly non-privacy content\n• Try disabling 'strict privacy filter' in sidebar\n• Try checking back later")
        st.stop()

    if len(articles) < 20:
        st.warning(f"Only found {len(articles)} privacy-relevant articles (target: 20). Try disabling strict filter for more results.")
    else:
        st.success(f"✅ Collected {len(articles)} privacy-focused articles (diversified across categories).")

    
    with st.spinner("제목 한국어로 번역 중..."):
        for a in articles:
            try:
                a.title = translate_title_ko(a.title)
            except Exception:
                pass

    # Generate Korean summaries (optional)
    with st.spinner("한국어 요약 생성 중..."):
        for a in articles:
            seed_text = getattr(a, "summary", None) or getattr(a, "description", None)
            try:
                a.summary = cached_korean_md(a.title, seed_text, a.source)
            except Exception:
                a.summary = None


    formatter = create_formatter(use_emoji=format_choice)

    if format_choice == "category":
        output_md = formatter.format_by_category(articles)
    elif format_choice == "simple":
        output_md = formatter.format_simple_list(articles)
    else:
        output_md = formatter.format_article_list(articles)

    st.subheader("Report")
    st.markdown(output_md)

    # Eye-catching download button with custom styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pdf_bytes = markdown_to_pdf(output_md)

        pdf_bytes = markdown_to_pdf(output_md)

        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_bytes,
            file_name="privacy_news_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

else:
    st.info("Click 🔄 Refresh to fetch the latest 20 categorized articles.")