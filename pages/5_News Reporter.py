import os
import json
import requests
import streamlit as st
from datetime import datetime, time
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("SONAR_API_KEY")
API_URL = "https://api.perplexity.ai/chat/completions"

# Trusted Vietnamese finance news sites
VN_SOURCES = [
    "vietstock.vn",
    "cafef.vn",
    "vnexpress.net",
    "ndh.vn",
    "vneconomy.vn",
    "nhipcaudautu.vn"
]

# ------------- Helper Functions ------------------

def fetch_news(tickers, domains=VN_SOURCES, recency="day", context_size="medium"):
    """Call Perplexity Sonar API and fetch Vietnamese news summaries about tickers."""
    if not API_KEY:
        st.error("❌ Missing SONAR_API_KEY. Please add it to your .env file.")
        return None, None, []

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Build enhanced search terms with both tickers and company names
    enhanced_search_terms = []
    for ticker in tickers:
        search_terms = get_search_terms(ticker)
        enhanced_search_terms.append(search_terms)
    
    # Build user prompt based on recency period
    if recency == "day":
        time_period = "2 ngày gần nhất"
        prompt = (
            f"Tìm TẤT CẢ tin tức về các công ty sau trong {time_period}: {', '.join(enhanced_search_terms)}. "
            f"Tìm kiếm cả MÃ CỔ PHIẾU và TÊN CÔNG TY để có kết quả chính xác nhất. "
            f"CHỈ báo cáo tin tức có chứa tên mã cổ phiếu hoặc tên công ty cụ thể. KHÔNG báo cáo tin tức chung về thị trường."
            f"Nếu không tìm thấy tin tức về công ty nào, bỏ qua công ty đó."
            f"Định dạng: Mỗi tin tức là MỘT DÒNG RIÊNG, bắt đầu bằng **MÃ CỔ PHIẾU**: theo sau là nội dung và nguồn."
            f"Tập trung vào: kết quả kinh doanh, thay đổi nhân sự, kế hoạch tăng vốn, thông báo quan trọng. "
            f"Trả lời bằng tiếng Việt."
        )
    else:  # week or other periods
        time_period = "2 tuần gần nhất" if recency == "week" else "thời gian gần đây"
        prompt = (
            f"Tìm TẤT CẢ tin tức về các công ty sau trong {time_period}: {', '.join(enhanced_search_terms)}. "
            f"Tìm kiếm cả MÃ CỔ PHIẾU và TÊN CÔNG TY để có kết quả chính xác nhất. "
            f"CHỈ báo cáo tin tức có chứa tên mã cổ phiếu hoặc tên công ty cụ thể. KHÔNG báo cáo tin tức chung về thị trường."
            f"Định dạng: **MÃ CỔ PHIẾU**: [Nội dung tin tức] - [Nguồn] "
            f"Tập trung vào: kết quả kinh doanh, thay đổi nhân sự, kế hoạch mới, biến động giá, tin ngành. "
            f"Trả lời bằng tiếng Việt."
        )

    # Add date-based seed for consistency within the same day
    import hashlib
    from datetime import datetime
    
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    tickers_str = ",".join(sorted(tickers))  # Sort for consistency
    seed_string = f"{current_date_str}-{tickers_str}-{recency}"
    seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16) % 1000
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {"role": "system", "content": "Bạn là phóng viên tài chính chuyên nghiệp. Hãy tìm và trả về các tin tức một cách NHẤT QUÁN. Luôn ghi rõ mã cổ phiếu ở đầu mỗi tin tức và ngày đăng bài báo ở cuối mỗi tin tức nếu có. Trả về kết quả ổn định và đáng tin cậy."},
            {"role": "user", "content": prompt}
        ],
        "search_recency_filter": recency,  # "day", "week", "month"
        "search_domain_filter": domains,   # restrict to VN finance websites
        "web_search_options": {"search_context_size": "high"},  # Increased for more comprehensive results
        "return_related_questions": False,
        "temperature": 0.1  # Lower temperature for more consistent results
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        summary = data["choices"][0]["message"]["content"]
        sources = data.get("search_results", [])
        
        # Simpler filtering - only remove obvious "no news" lines, keep more content
        lines = summary.split('\n')
        filtered_lines = []
        found_tickers = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Only skip very specific "no news" patterns
            skip_patterns = [
                "không có tin tức cụ thể về các mã cổ phiếu",
                "không tìm thấy tin tức nào về",
                "không có thông tin mới về"
            ]
            
            should_skip = any(pattern.lower() in line.lower() for pattern in skip_patterns)
            
            if not should_skip:
                # Display each line directly to the app
                st.write(line)
                filtered_lines.append(line)
                # Check for ticker mentions
                for ticker in tickers:
                    if ticker.upper() in line.upper() and ticker not in found_tickers:
                        found_tickers.append(ticker)
        
        # Simple filtering - keep most content, minimal processing
        filtered_summary = '\n'.join(filtered_lines) if filtered_lines else "Không có tin tức mới trong khoảng thời gian được yêu cầu."
        
        # Ensure each ticker starts on a new line
        formatted_summary = filtered_summary
        for ticker in tickers:
            # Replace pattern where ticker appears after other content on same line
            formatted_summary = formatted_summary.replace(f' {ticker}:', f'\n\n**{ticker}**:')
            formatted_summary = formatted_summary.replace(f' **{ticker}**:', f'\n\n**{ticker}**:')
        
        # Clean up any double line breaks at the beginning
        formatted_summary = formatted_summary.strip()
        
        return formatted_summary, sources, found_tickers
    except Exception as e:
        st.error(f"API error: {e}")
        return None, None, []

# Priority tickers for brokerage and banking industry (in priority order)
PRIORITY_TICKERS = ["SSI", "VND", "VCI", "HCM", "VIX", "SHS"]
OTHER_TICKERS = ["BSI", "FTS", "VIG", "CTS", "TCB", "VCB", "BID", "CTG", "MBB", "VPB", 
                 "TPB", "STB", "ACB", "HDB", "MSB", "EIB", "VIB", "SHB", "OCB", "LPB", "NAB", "KLB", "BVB"]
ALL_PREDEFINED_TICKERS = PRIORITY_TICKERS + OTHER_TICKERS

# Ticker to Company Name Mapping for Enhanced Search Accuracy
TICKER_TO_COMPANY = {
    # Priority Brokerage Companies
    "SSI": "SSI",
    "VND": "VNDirect", 
    "VCI": "Vietcap",
    "HCM": "HSC",
    "VIX": "VIX Securities",
    "SHS": "Saigon Hanoi Securities",
    "IPA": "I.P.A",
    
    # Other Brokerage Companies
    "BSI": "BSI",
    "FTS": "Chứng khoán FPT", 
    "CTS": "Chứng khoán Công Thương",
    
    # Major Banks
    "TCB": "Techcombank",
    "VCB": "Vietcombank", 
    "BID": "BIDV",
    "CTG": "Vietinbank",
    "MBB": "Ngân hàng Quân Đội",
    "VPB": "VPBank",
    "TPB": "Ngân hàng Tiên Phong",
    "STB": "Sacombank",
    "ACB": "Ngân hàng Á Châu",
    "HDB": "HDBank",
    "MSB": "Ngân hàng Maritime",
    "EIB": "Eximbank",
    "VIB": "VIB Bank",
    "SHB": "SHB Bank", 
    "OCB": "Ngân hàng Phương Đông",
    "LPB": "Ngân hàng Lộc Phát",
    "NAB": "Nam Á",
    "KLB": "Ngân hàng Kiên Long",
    "BVB": "Ngân hàng Bảo Việt"
}

def get_search_terms(ticker):
    """Get both ticker and company name for enhanced search accuracy."""
    company_name = TICKER_TO_COMPANY.get(ticker, "")
    if company_name:
        return f"{ticker} OR {company_name}"
    return ticker

# ------------- Streamlit UI ------------------

st.set_page_config(page_title="📰 Vietnam Finance News Reporter", layout="wide")
st.title("📰 News Reporter")

# Initialize session state for news
if 'banking_news_loaded' not in st.session_state:
    st.session_state.banking_news_loaded = False
if 'banking_summary' not in st.session_state:
    st.session_state.banking_summary = None
if 'banking_sources' not in st.session_state:
    st.session_state.banking_sources = None
if 'banking_found_tickers' not in st.session_state:
    st.session_state.banking_found_tickers = []
if 'last_news_date' not in st.session_state:
    st.session_state.last_news_date = None

# Brokerage & Banking News Alert (moved to top)
st.subheader("🏦 Brokerage & Banking News Alert")

# Manual refresh button only
col1, col2 = st.columns([1, 4])
with col1:
    load_banking_news = st.button("📰 Load Latest News")

if load_banking_news:
    with st.spinner("Fetching all news from the last 2 days..."):
        # Always get fresh results from last 2 days (no caching, no session state dependency)
        predefined_summary, predefined_sources, found_tickers = fetch_news(ALL_PREDEFINED_TICKERS, recency="day")
    
    # Store in session state
    st.session_state.banking_summary = predefined_summary
    st.session_state.banking_sources = predefined_sources
    st.session_state.banking_found_tickers = found_tickers
    st.session_state.banking_news_loaded = True

# Always display news if available
if st.session_state.banking_summary:
    with st.container():
        found_tickers_str = ", ".join(st.session_state.banking_found_tickers) if st.session_state.banking_found_tickers else "không xác định"
        st.success(f"✅ Latest news for brokerage and banking stocks (Found: {found_tickers_str}):")
        # News is already displayed directly in fetch_news() function
        
        if st.session_state.banking_sources:
            with st.expander("📋 Detailed Sources"):
                for s in st.session_state.banking_sources:
                    title = s.get("title", "(no title)")
                    url = s.get("url", "#")
                    date = s.get("date", "")
                    st.markdown(f"- [{title}]({url}) ({date})")
        
elif st.session_state.banking_news_loaded:
    st.info("ℹ️ No banking/brokerage news found in the last 2 days.")
else:
    st.info("💡 Click 'Load Latest News' to fetch the latest news from the past 2 days.")

st.divider()
st.subheader("🔍 Company News Search")

tickers = st.text_input("Enter tickers:", "")
tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]

col1, col2 = st.columns([1,1])
with col1:
    refresh = st.button("🔄 Refresh News")

with col2:
    st.caption("⏰ The app will automatically refresh at **10:00 AM** every day.")

# Remove caching to ensure fresh results every time
# @st.cache_data(ttl=60*60)  # cache 1 hour - DISABLED for consistency
def get_fresh_news(tickers):
    return fetch_news(tickers)

# Custom ticker news search (only show when user enters tickers)
if tickers_list:
    # Always get fresh news for custom tickers using 2-week period
    summary, sources, found_tickers = fetch_news(tickers_list, recency="week")

    # Display custom ticker news results
    if summary:
        st.subheader("📰 Custom Ticker News")
        if found_tickers:
            st.success(f"✅ Found news for: {', '.join(found_tickers)}")
        else:
            st.warning("⚠️ No specific ticker mentions found - results may be generic")
        # News is already displayed directly in fetch_news() function

    # Show sources for custom tickers
    if sources:
        st.subheader("🔗 News Sources")
        for s in sources:
            title = s.get("title", "(no title)")
            url = s.get("url", "#")
            date = s.get("date", "")
            st.markdown(f"- [{title}]({url}) ({date})")
    else:
        st.info("No news sources found for the entered tickers.")
