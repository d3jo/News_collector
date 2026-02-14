import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from textwrap import wrap
import time


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))



from privacy_news_algorithm.collector import create_collector, Article
from privacy_news_algorithm.formatter import create_formatter
from privacy_news_algorithm.summarizer import summarize_korean_bullets

# Import Sentence-BERT deduplication
try:
    from dedup import (
        get_sbert_deduplicator,
        SBERT_AVAILABLE
    )
    DEDUP_AVAILABLE = SBERT_AVAILABLE
except ImportError:
    DEDUP_AVAILABLE = False

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


st.set_page_config(
    page_title="Privacy News Report", 
    page_icon="🛡️", 
    layout="wide")

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Define color schemes
if st.session_state.dark_mode:
    bg_color = "#0e1117"
    bg_color_opp = "#ffffff"
    secondary_bg = "#262730"
    text_color = "#fafafa"
    text_color_opp = "#262730"
    border_color = "#444"
    card_bg = "#1e1e1e"
    header_color = "#fafafa"
    input_bg = "#262730"
    input_text = "#fafafa"
else:
    bg_color = "#ffffff"
    bg_color_opp = "#0e1117"
    secondary_bg = "#f0f2f6"
    text_color = "#262730"
    text_color_opp = "#fafafa"
    border_color = "#e0e0e0"
    card_bg = "#ffffff"
    header_color = "#262730"
    input_bg = "#ffffff"
    input_text = "#262730"

# [CSS styles - keeping same as before]
st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; }}
    header[data-testid="stHeader"] {{ background-color: {bg_color} !important; }}
    .stApp > header {{ background-color: {bg_color_opp} !important; }}
    .main .block-container {{ background-color: {bg_color} !important; }}
    .stApp p, .stApp span, .stApp div {{ color: {text_color} !important; }}
    [data-testid="stSidebar"] {{ background-color: {secondary_bg}; }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div, [data-testid="stSidebar"] label {{ color: {text_color} !important; }}
    .stMarkdown {{ color: {text_color} !important; }}
    .stMarkdown p, .stMarkdown span, .stMarkdown li {{ color: {text_color} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {header_color} !important; }}
    .stButton > button[kind="primary"] {{ color: white !important; }}
    .stButton > button:not([kind="primary"]) {{ color: {input_text} !important; background-color: {input_bg} !important; border: 1px solid {border_color} !important; }}
    [data-testid="stSidebar"] .stButton > button {{ background-color: {input_bg} !important; color: {input_text} !important; border: 1px solid {border_color} !important; }}
    input, textarea {{ color: {input_text} !important; background-color: {input_bg} !important; border: 1px solid {border_color} !important; }}
    select, .stSelectbox select {{ color: {input_text} !important; background-color: {input_bg} !important; border: 1px solid {border_color} !important; }}
    option {{ color: {input_text} !important; background-color: {input_bg} !important; }}
    [data-baseweb="select"] {{ background-color: {input_bg} !important; }}
    [data-baseweb="select"] span, [data-baseweb="select"] div {{ color: {input_text} !important; }}
    .stCaption {{ color: {text_color} !important; opacity: 0.7; }}
    .stAlert {{ background-color: {card_bg} !important; border: 1px solid {border_color} !important; color: {text_color} !important; }}
    .stAlert p, .stAlert div {{ color: {text_color} !important; }}
    .stDownloadButton > button {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; color: white !important; font-size: 18px !important; font-weight: bold !important; padding: 16px 32px !important; border-radius: 12px !important; border: none !important; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important; transition: all 0.3s ease !important; }}
    .stDownloadButton > button:hover {{ transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important; }}
    .stCheckbox label, .stRadio label {{ color: {text_color} !important; }}
    .stCheckbox span, .stRadio span {{ color: {text_color} !important; }}
    .stSelectbox label {{ color: {text_color} !important; }}
    .stSelectbox > div > div {{ color: {input_text} !important; background-color: {input_bg} !important; }}
    [role="listbox"] {{ background-color: {input_bg} !important; }}
    [role="option"] {{ color: {input_text} !important; background-color: {input_bg} !important; }}
    [role="option"]:hover {{ background-color: {border_color} !important; }}
    code {{ background-color: {secondary_bg} !important; color: {text_color} !important; }}
    a {{ color: #667eea !important; }}
    .stSpinner > div {{ border-top-color: #667eea !important; }}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 실시간 개인정보 보안 동향 모니터링 리포트")
st.caption("Refresh to fetch the latest privacy-related news and categorize it.")

# --- Sidebar ---
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

# Theme
st.sidebar.subheader("🎨 Theme")
mode_text = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
st.sidebar.write(f"Current: **{mode_text}**")
theme_button_label = "Switch to ☀️ Light Mode" if st.session_state.dark_mode else "Switch to 🌙 Dark Mode"
if st.sidebar.button(theme_button_label, use_container_width=True):
    toggle_theme()
    st.rerun()

st.sidebar.divider()

# Deduplication settings (hidden but active)
# SBERT deduplication runs automatically with optimal settings
if DEDUP_AVAILABLE:
    use_sbert_dedup = True  # Always enabled
    similarity_threshold = 0.82  # Optimal balanced setting
else:
    use_sbert_dedup = False
    similarity_threshold = 0.82


format_choice = st.sidebar.selectbox("Output format", ["category", "simple", "detailed"], index=0)

# Main controls
st.subheader("Latest Privacy Monitor")
st.caption("One click. Latest privacy news. Categorized into 정책/규제, 사건/조사, 신기술, 기타.")

run = st.button("🔄 Refresh", type="primary")

MONITOR_QUERY = '"data privacy" OR "privacy law" OR "privacy regulation" OR "privacy breach" OR GDPR OR privacy OR "privacy policy" OR "privacy violation" OR "privacy protection" OR "data protection" OR "personal data" OR "consumer privacy" OR "privacy rights" OR "user privacy" OR "information privacy"'
QUERY_NEWSAPI = MONITOR_QUERY
QUERY_SIMPLE = "data privacy OR privacy law OR GDPR OR privacy breach OR data protection OR consumer privacy"


def markdown_to_pdf(md_text: str) -> bytes:
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    x_margin = 1 * inch
    y_margin = 1 * inch
    y = height - y_margin
    c.setFont("HYSMyeongJo-Medium", 10)

    for line in md_text.split("\n"):
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
    """Basic URL/title deduplication"""
    seen_urls = set()
    seen_titles = set()
    out = []
    
    for a in sorted(articles, key=lambda x: x.published_date or "", reverse=True):
        url = (a.url or "").strip().lower().split('?')[0].rstrip('/')
        title = " ".join((a.title or "").lower().split())
        
        if not url and not title:
            continue
        if url and url in seen_urls:
            continue
        if title and title in seen_titles:
            continue
        
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        out.append(a)
    
    return out


def dedupe_with_sbert_titles(
    articles: list[Article],
    similarity_threshold: float = 0.82,
    progress_callback=None
) -> tuple[list[Article], int]:
    """
    TITLE-BASED SBERT deduplication on ENGLISH titles
    
    CRITICAL: This runs BEFORE translation!
    """
    if not DEDUP_AVAILABLE:
        return dedupe_keep_latest(articles), 0
    
    if not articles:
        return [], 0
    
    # Log initial state
    if progress_callback:
        progress_callback(f"🔍 시작: {len(articles)}개 기사 (영문 원본)")
    
    # Step 1: Basic dedupe
    if progress_callback:
        progress_callback("1️⃣ 키워드 기반 중복 제거중 (URL/exact title)...")
    
    articles = dedupe_keep_latest(articles)
    after_basic = len(articles)
    
    if progress_callback:
        progress_callback(f"1️⃣ 완료: {after_basic}개 남음")
    
    # Step 2: SBERT title embedding
    if progress_callback:
        progress_callback("2️⃣ AI 제목 벡터 임베딩 생성 중...")
    
    try:
        deduplicator = get_sbert_deduplicator()
        
        embeddings = []
        for idx, article in enumerate(articles):
            # Use ENGLISH title + description snippet
            desc_snippet = ""
            if hasattr(article, 'summary') and article.summary:
                desc_snippet = article.summary[:100]
            elif hasattr(article, 'description') and article.description:
                desc_snippet = article.description[:100]
            
            text_to_embed = f"{article.title}. {desc_snippet}".strip()
            embedding = deduplicator.create_embedding(text_to_embed)
            embeddings.append(embedding)
            
            if idx % 5 == 0 and progress_callback:
                progress_callback(f"2️⃣ 임베딩: {idx}/{len(articles)}")
        
        if progress_callback:
            progress_callback("3️⃣ 유사도 분석 및 중복 제거 중...")
        
        # Step 3: Similarity-based deduplication
        keep_indices = []
        removed_items = []
        
        for i in range(len(articles)):
            if i % 20 == 0 and progress_callback:
                progress_callback(f"3️⃣ 분석: {i}/{len(articles)}")
            
            is_duplicate = False
            
            for j in keep_indices:
                if embeddings[i] is None or embeddings[j] is None:
                    continue
                
                # Adjust threshold for same source + same day
                effective_threshold = similarity_threshold
                if articles[i].source == articles[j].source:
                    if (hasattr(articles[i], 'published_date') and 
                        hasattr(articles[j], 'published_date') and
                        articles[i].published_date and articles[j].published_date):
                        
                        try:
                            date_i = articles[i].published_date.date() if hasattr(articles[i].published_date, 'date') else articles[i].published_date
                            date_j = articles[j].published_date.date() if hasattr(articles[j].published_date, 'date') else articles[j].published_date
                            
                            if date_i == date_j:
                                # 15% more lenient for same source + day
                                effective_threshold = similarity_threshold * 0.85
                        except:
                            pass
                
                similarity = deduplicator.cosine_similarity(embeddings[i], embeddings[j])
                
                if similarity >= effective_threshold:
                    is_duplicate = True
                    removed_items.append({
                        "kept_title": articles[j].title,
                        "removed_title": articles[i].title,
                        "similarity": similarity,
                        "threshold": effective_threshold,
                        "source": articles[i].source
                    })
                    break
            
            if not is_duplicate:
                keep_indices.append(i)
        
        deduplicated = [articles[i] for i in keep_indices]
        num_removed = after_basic - len(deduplicated)
        
        # Log results
        if progress_callback:
            progress_callback(f"✅ AI 중복 제거 완료: {num_removed}개 제거")
            
        
        return deduplicated, num_removed
        
    except Exception as e:
        st.error(f"SBERT 오류: {e}")
        import traceback
        st.code(traceback.format_exc())
        return articles, 0


def is_privacy_relevant(article: Article) -> bool:
    """Privacy relevance filter"""
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

    score = 0
    score += sum(3 for t in strong if t in title)
    score += sum(1 for t in strong if t in body)
    score += sum(2 for t in adjacent if t in title)
    score += sum(1 for t in adjacent if t in body)

    return score >= 3


def pick_20_diverse(articles: list[Article], per_cat: int = 5, total: int = 20) -> list[Article]:
    """Pick diverse articles across categories"""
    ordered = sorted(articles, key=lambda x: x.published_date or "", reverse=True)

    buckets = {"policy": [], "incident": [], "technology": [], "general": []}
    seen_urls = set()
    seen_titles = set()

    for a in ordered:
        url = (a.url or "").strip().lower().split('?')[0].rstrip('/')
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

    for a in ordered:
        if len(out) >= total:
            break
        url = (a.url or "").strip().lower().split('?')[0].rstrip('/')
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
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator. Translate headlines to Korean, keeping proper nouns in English when commonly used. Output ONLY the translated headline."},
                {"role": "user", "content": f"Translate: {title}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation failed: {e}")
        return title


@st.cache_data(show_spinner=False)
def cached_korean_md(title, desc, source):
    try:
        bullets = summarize_korean_bullets(title, desc, source)
        return "\n".join([f"- {b}" for b in bullets])
    except Exception as e:
        print(f"Summarization failed: {e}")
        return "(요약 생성 실패)"


if run:
    collector = create_collector(
        gnews_api_key=os.getenv("GNEWS_API_KEY"),
        newsapi_key=os.getenv("NEWSAPI_KEY"),
    )

    # === STEP 1: COLLECT (English titles) ===
    with st.spinner("📰 Collecting latest articles..."):
        articles = []
        articles.extend(collector.collect_from_gnews(QUERY_SIMPLE))
        time.sleep(1)
        articles.extend(collector.collect_from_newsapi(QUERY_NEWSAPI, max_pages=3))
        time.sleep(1)
        articles.extend(collector.collect_from_mediastack(QUERY_SIMPLE, max_pages=4))
        time.sleep(1)
        articles.extend(collector.collect_from_currents(QUERY_SIMPLE, max_pages=2))
        time.sleep(1)
        articles.extend(collector.collect_from_newsdata(QUERY_SIMPLE, max_pages=3))
        
        st.info(f"📊 수집 완료: {len(articles)}개 기사")

    if not articles:
        st.warning("No articles found.")
        st.stop()

    # === STEP 2: DEDUPLICATE (BEFORE translation!) ===
    progress_placeholder = st.empty()
    
    def update_progress(msg):
        progress_placeholder.info(msg)
    
    
    if use_sbert_dedup and DEDUP_AVAILABLE:
        with st.spinner("🤖 AI 기반 의미 분석 중복 제거..."):
            articles, num_removed = dedupe_with_sbert_titles(
                articles,
                similarity_threshold=similarity_threshold,
                progress_callback=update_progress
            )
    else:
        with st.spinner("기본 중복 제거..."):
            before_count = len(articles)
            articles = dedupe_keep_latest(articles)
            num_removed = before_count - len(articles)
            st.info(f"기본 중복 제거: {num_removed}개 제거 → {len(articles)}개 남음")
    
    # === STEP 3: FILTER ===
    # === STEP 3: FILTER ===
    with st.spinner("🔍 개인정보 관련성 필터링 중..."):
        before_filter = len(articles)
        privacy_articles = [a for a in articles if is_privacy_relevant(a)]
        after_filter = len(privacy_articles)
        articles = privacy_articles

    if not articles:
        st.warning("No privacy-relevant articles found.")
        st.stop()

    # === STEP 4: SELECT DIVERSE ===
    with st.spinner("📑 카테고리별 분산 선택..."):
        articles = pick_20_diverse(articles, per_cat=5, total=20)
        st.success(f"✅ 최종 선택: {len(articles)}개 기사")

    # === STEP 5: TRANSLATE (AFTER dedup!) ===
    st.info("🌐 한국어 번역 시작 (중복 제거 후)")
    with st.spinner("번역 중..."):
        for i, a in enumerate(articles):
            try:
                a.title = translate_title_ko(a.title)
                time.sleep(0.2)
            except Exception as e:
                print(f"Translation error: {e}")

    # === STEP 6: SUMMARIZE ===
    with st.spinner("📝 한국어 요약 생성..."):
        for i, a in enumerate(articles):
            seed_text = getattr(a, "summary", None) or getattr(a, "description", None)
            try:
                summary = cached_korean_md(a.title, seed_text, a.source)
                a.summary = summary
            except Exception as e:
                print(f"Summarization error: {e}")
                a.summary = "(요약 없음)"

    # === DISPLAY ===
    formatter = create_formatter(use_emoji=format_choice)

    if format_choice == "category":
        output_md = formatter.format_by_category(articles)
    elif format_choice == "simple":
        output_md = formatter.format_simple_list(articles)
    else:
        output_md = formatter.format_article_list(articles)

    st.subheader("Report")
    st.markdown(output_md)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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