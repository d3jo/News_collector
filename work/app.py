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


st.set_page_config(
    page_title="Privacy News Report", 
    page_icon="🛡️", 
    layout="wide")

# Initialize theme in session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True  # Default to dark mode

# Theme toggle function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# Define color schemes
if st.session_state.dark_mode:
    # Dark mode colors
    bg_color = "#0e1117"
    bg_color_opp = "#ffffff"
    secondary_bg = "#262730"
    text_color = "#fafafa"
    text_color_opp = "#262730"
    border_color = "#444"
    card_bg = "#1e1e1e"
    header_color = "#fafafa"
    # Special colors for inputs/buttons in dark mode
    input_bg = "#262730"
    input_text = "#fafafa"
else:
    # Light mode colors
    bg_color = "#ffffff"
    bg_color_opp = "#0e1117"
    secondary_bg = "#f0f2f6"
    text_color = "#262730"
    text_color_opp = "#fafafa"
    border_color = "#e0e0e0"
    card_bg = "#ffffff"
    header_color = "#262730"
    # Special colors for inputs/buttons in light mode
    input_bg = "#ffffff"
    input_text = "#262730"

# Custom CSS with theme support
st.markdown(f"""
<style>
    /* Global theme colors */
    .stApp {{
        background-color: {bg_color};
    }}
    
    /* Fix top header area */
    header[data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    
    /* Toolbar at the top */
    .stApp > header {{
        background-color: {bg_color_opp} !important;
    }}
    
    /* Main content area */
    .main .block-container {{
        background-color: {bg_color} !important;
    }}
    
    /* Main content text */
    .stApp p, .stApp span, .stApp div {{
        color: {text_color} !important;
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {secondary_bg};
    }}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {{
        color: {text_color} !important;
    }}
    
    /* Markdown content */
    .stMarkdown {{
        color: {text_color} !important;
    }}
    
    .stMarkdown p, .stMarkdown span, .stMarkdown li {{
        color: {text_color} !important;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {header_color} !important;
    }}
    
    /* Buttons - primary buttons get white text, sidebar buttons get appropriate color */
    .stButton > button[kind="primary"] {{
        color: white !important;
    }}
    
    .stButton > button:not([kind="primary"]) {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Sidebar buttons need dark text in light mode, light text in dark mode */
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Input fields and text areas */
    input, textarea {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Select dropdown - fix the white on white issue */
    select, .stSelectbox select {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    
    /* Dropdown options */
    option {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
    }}
    
    /* Fix for Streamlit's custom select widget */
    [data-baseweb="select"] {{
        background-color: {input_bg} !important;
    }}
    
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {{
        color: {input_text} !important;
    }}
    
    /* Caption and small text */
    .stCaption {{
        color: {text_color} !important;
        opacity: 0.7;
    }}
    
    /* Info/Warning/Success boxes */
    .stAlert {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        color: {text_color} !important;
    }}
    
    .stAlert p, .stAlert div {{
        color: {text_color} !important;
    }}
    
    /* Download button styling */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 16px 32px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }}
    
    .stDownloadButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }}
    
    .stDownloadButton > button:active {{
        transform: translateY(0px) !important;
    }}
    
    /* Checkbox and radio labels */
    .stCheckbox label, .stRadio label {{
        color: {text_color} !important;
    }}
    
    .stCheckbox span, .stRadio span {{
        color: {text_color} !important;
    }}
    
    /* Selectbox label and text */
    .stSelectbox label {{
        color: {text_color} !important;
    }}
    
    .stSelectbox > div > div {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
    }}
    
    /* Selectbox dropdown menu */
    [role="listbox"] {{
        background-color: {input_bg} !important;
    }}
    
    [role="option"] {{
        color: {input_text} !important;
        background-color: {input_bg} !important;
    }}
    
    [role="option"]:hover {{
        background-color: {border_color} !important;
    }}
    
    /* Code blocks */
    code {{
        background-color: {secondary_bg} !important;
        color: {text_color} !important;
    }}
    
    /* Links */
    a {{
        color: #667eea !important;
    }}
    
    /* Spinner */
    .stSpinner > div {{
        border-top-color: #667eea !important;
    }}
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

# Theme display and toggle
st.sidebar.subheader("🎨 Theme")
mode_text = "🌙 Dark Mode" if st.session_state.dark_mode else "☀️ Light Mode"
st.sidebar.write(f"Current: **{mode_text}**")

# Theme toggle button
theme_button_label = "Switch to ☀️ Light Mode" if st.session_state.dark_mode else "Switch to 🌙 Dark Mode"
if st.sidebar.button(theme_button_label, use_container_width=True):
    toggle_theme()
    st.rerun()

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
    return score >= 3

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
    """✅ FIXED: Use correct OpenAI API"""
    if not title:
        return title
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator that converts English news headlines to natural Korean."},
                {"role": "user", "content": f"""Translate this news headline into natural Korean.
Rules:
- Keep proper nouns (company/product/person names) in English if commonly used.
- Keep it short like a real Korean headline.
- Output ONLY the translated headline.

Headline: {title}"""}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return title  # Return original title if translation fails



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
        articles.extend(collector.collect_from_gnews(QUERY_SIMPLE))
        articles.extend(collector.collect_from_newsapi(QUERY_NEWSAPI, max_pages=3))
        articles.extend(collector.collect_from_mediastack(QUERY_SIMPLE, max_pages=4))
        articles.extend(collector.collect_from_currents(QUERY_SIMPLE, max_pages=5))
        articles.extend(collector.collect_from_newsdata(QUERY_SIMPLE, max_pages=10))
        
        # Deduplicate first
        articles = dedupe_keep_latest(articles)
        
        # Filter for privacy relevance (optional)
        privacy_articles = [a for a in articles if is_privacy_relevant(a)]
        st.info(f"Found {len(articles)} articles, {len(privacy_articles)} are privacy-relevant (strict filter enabled)")
        articles = privacy_articles

        # Pick diverse set from articles
        articles = pick_20_diverse(articles, per_cat=5, total=20)

    if not articles:
        st.warning("No privacy-relevant articles found. This could mean:")
        st.info("• Not enough recent privacy news in the past 48 hours\n• API returned mostly non-privacy content\n• Try disabling 'strict privacy filter' in sidebar\n• Try checking back later")
        st.stop()


    st.info(f"Found {len(articles)} privacy-relevant articles")


    with st.spinner("제목 한국어로 번역 중..."):
        for a in articles:
            try:
                a.title = translate_title_ko(a.title)
            except Exception as e:
                print(f"Translation failed for article: {e}")
                pass

    # Generate Korean summaries (optional)

    with st.spinner("한국어 요약 생성 중..."):
        for a in articles:
            seed_text = getattr(a, "summary", None) or getattr(a, "description", None)
            try:
                a.summary = cached_korean_md(a.title, seed_text, a.source)
            except Exception as e:
                print(f"Summary generation failed: {e}")
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