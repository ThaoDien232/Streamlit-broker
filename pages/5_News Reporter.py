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
    "cafef.vn",
    "vnexpress.net",
    "vneconomy.vn",
    "fireant.vn",
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
            f"Tìm TẤT CẢ tin tức về các công ty sau trong {time_period}: {', '.join(tickers)}. "
            f"Tìm kiếm cả MÃ CỔ PHIẾU và TÊN CÔNG TY để có kết quả chính xác nhất. "
            f"CHỈ báo cáo tin tức có chứa tên mã cổ phiếu hoặc tên công ty cụ thể. "
            f"CHỈ báo cáo tin tức nếu có bài báo hoặc thông báo CỤ THỂ từ các nguồn báo chí chính thống "
            f"(CafeF, VnExpress, VnEconomy, hoặc công bố thông tin trên website công ty). "
            f"TUYỆT ĐỐI KHÔNG được tự suy đoán hoặc tạo ra tin tức nếu không tìm thấy bài báo. "
            f"TUYỆT ĐỐI BỎ QUA các tin tức không liên quan đến chủ đề cụ thể sau: "
            f"- biến động giá cổ phiếu (tăng, giảm, % thay đổi, mức giá, vốn hóa), "
            f"- giao dịch khối ngoại, tự doanh, dòng tiền, khối lượng giao dịch, "
            f"- phân tích kỹ thuật, khuyến nghị đầu tư, tâm lý thị trường, "
            f"- các chủ đề xã hội, chính trị, kinh tế vĩ mô không liên quan trực tiếp đến công ty, "
            f"- tin tức về ngành nghề nói chung mà không nêu tên công ty cụ thể. "
            f"ƯU TIÊN VÀ BẮT BUỘC GIỮ LẠI các tin tức quan trọng về: "
            f"- kết quả kinh doanh (doanh thu, lợi nhuận, báo cáo tài chính, kết quả quý/năm), "
            f"- thay đổi nhân sự cấp cao (bổ nhiệm, từ nhiệm, thay thế CEO/lãnh đạo), "
            f"- tất cả hoạt động tăng vốn: phát hành cổ phiếu riêng lẻ, chào bán cổ phiếu cho cổ đông hiện hữu, IPO, "
            f"- phát hành trái phiếu, huy động vốn, tăng vốn điều lệ, "
            f"- hoạt động M&A (sáp nhập, mua bán, đầu tư chiến lược), "
            f"- dự án đầu tư lớn, mở rộng kinh doanh, thành lập công ty con, "
            f"- hợp tác chiến lược, ký kết hợp đồng lớn, "
            f"- vướng mắc pháp lý, xử phạt, điều tra của cơ quan quản lý, "
            f"- thông báo từ ĐHCĐ, quyết định của HĐQT, nghị quyết quan trọng. "
            f"QUAN TRỌNG: KHÔNG BỎ SÓT bất kỳ tin tức nào về 'chào bán cổ phiếu riêng lẻ', 'private placement', 'tăng vốn', 'huy động vốn'. "
            f"Nếu KHÔNG có tin tức nào phù hợp cho một công ty, HOÀN TOÀN BỎ QUA công ty đó (không viết dòng 'không có tin tức'). "
            f"Định dạng: Mỗi tin tức là MỘT DÒNG RIÊNG, bắt đầu bằng **MÃ CỔ PHIẾU**: theo sau là nội dung và LUÔN KẾT THÚC BẰNG LINK TRỰC TIẾP đến bài báo hoặc thông tin gốc. "
            f"Trả lời bằng tiếng Việt."
        )
    else:  # week or other periods
        time_period = "2 tuần gần nhất" if recency == "week" else "thời gian gần đây"

        prompt = (
            f"Tìm TẤT CẢ tin tức về các công ty sau trong {time_period}: {', '.join(tickers)}. "
            f"Tìm kiếm cả MÃ CỔ PHIẾU và TÊN CÔNG TY để có kết quả chính xác nhất. "
            f"CHỈ báo cáo tin tức có chứa tên mã cổ phiếu hoặc tên công ty cụ thể. "
            f"TUYỆT ĐỐI BỎ QUA các tin tức không liên quan đến chủ đề cụ thể sau: "
            f"- biến động giá cổ phiếu (tăng, giảm, % thay đổi, mức giá, vốn hóa), "
            f"- giao dịch khối ngoại, tự doanh, dòng tiền, khối lượng giao dịch, "
            f"- phân tích kỹ thuật, khuyến nghị đầu tư, tâm lý thị trường, "
            f"- các hoạt động PR, marketing, sự kiện triển lãm, hội chợ, tài trợ, giải thưởng, CSR, "
            f"- các chủ đề xã hội, chính trị, kinh tế vĩ mô không liên quan trực tiếp đến công ty, "
            f"- tin tức về ngành nghề nói chung mà không nêu tên công ty cụ thể. "
            f"ƯU TIÊN VÀ BẮT BUỘC GIỮ LẠI các tin tức quan trọng về: "
            f"- kết quả kinh doanh (doanh thu, lợi nhuận, báo cáo tài chính, kết quả quý/năm), "
            f"- thay đổi nhân sự cấp cao (bổ nhiệm, từ nhiệm, thay thế CEO/lãnh đạo), "
            f"- tất cả hoạt động tăng vốn: phát hành cổ phiếu riêng lẻ, chào bán cổ phiếu cho cổ đông hiện hữu, IPO, "
            f"- phát hành trái phiếu, huy động vốn, tăng vốn điều lệ, "
            f"- hoạt động M&A (sáp nhập, mua bán, đầu tư chiến lược), "
            f"- dự án đầu tư lớn, mở rộng kinh doanh, thành lập công ty con, "
            f"- hợp tác chiến lược, ký kết hợp đồng lớn, "
            f"- vướng mắc pháp lý, xử phạt, điều tra của cơ quan quản lý, "
            f"- thông báo từ ĐHCĐ, quyết định của HĐQT, nghị quyết quan trọng. "
            f"QUAN TRỌNG: KHÔNG BỎ SỐT bất kỳ tin tức nào về 'chào bán cổ phiếu riêng lẻ', 'private placement', 'tăng vốn', 'huy động vốn'. "
            f"Nếu không tìm thấy tin tức phù hợp cho công ty nào, bỏ qua công ty đó. "
            f"Định dạng: Mỗi tin tức là MỘT DÒNG RIÊNG, bắt đầu bằng **MÃ CỔ PHIẾU**: theo sau là nội dung và nguồn. "
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
            {"role": "system", "content": "Bạn là phóng viên tài chính chuyên nghiệp."
             "Chỉ báo cáo tin tức NẾU VÀ CHỈ NẾU tìm thấy bài báo hoặc thông báo gốc trong search_results. " 
             "Chỉ báo cáo tin tức LIÊN QUAN TRỰC TIẾP đến các mã cổ phiếu hoặc công ty được yêu cầu."
             "TUYỆT ĐỐI KHÔNG được tự suy đoán hoặc tạo ra tin tức. "
             "Hãy tìm và trả về các tin tức một cách TOÀN DIỆN và NHẤT QUÁN, đặc biệt chú ý đến:" 
             "- Các hoạt động tăng vốn, phát hành cổ phiếu riêng lẻ, chào bán cho cổ đông"
             "- Kết quả kinh doanh, báo cáo tài chính, thông báo từ ĐHCĐ và HĐQT"
             "- Thay đổi nhân sự, M&A, đầu tư chiến lược"
             "Luôn ghi rõ mã cổ phiếu ở đầu mỗi tin tức và ngày đăng bài báo ở cuối mỗi tin tức nếu có. "
             "Trả về kết quả ổn định và đáng tin cậy, KHÔNG BỎ SÓT tin tức quan trọng."
             "Rất quan trọng: chỉ liệt kê NGUỒN nào đã được sử dụng trực tiếp để viết tin tức trong phần trả lời. "
             "KHÔNG hiển thị nguồn không liên quan, không khớp với mã cổ phiếu."},
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
                # Don't display directly - let calling function handle display
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
PRIORITY_TICKERS = ["SSI", "VND", "VCI", "HCM", "VIX", "SHS", "IPA"]
OTHER_TICKERS = ["VCB", "BID", "VPB", 
                 "STB", "VIB", "SHB", "OCB"]
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
    
    # Major Banks
    "NHNN": "Ngân hàng nhà nước",
    "VCB": "Vietcombank", 
    "BID": "BIDV",
    "VPB": "VPBank",
    "STB": "Sacombank",
    "VIB": "VIB Bank",
    "SHB": "SHB Bank", 
    "OCB": "Ngân hàng Phương Đông",
}

def get_search_terms(ticker):
    """Get both ticker and company name for enhanced search accuracy."""
    company_name = TICKER_TO_COMPANY.get(ticker, "")
    if company_name:
        return f"{ticker} OR {company_name}"
    return ticker

def filter_relevant_sources(sources, news_content, tickers):
    """Filter sources to only include ones that are relevant to the displayed news content."""
    if not sources or not news_content:
        return []
    
    relevant_sources = []
    news_lower = news_content.lower()
    
    for source in sources:
        title = source.get("title", "").lower()
        url = source.get("url", "").lower()
        
        # Check if source mentions any of our tickers
        ticker_mentioned = any(ticker.lower() in title or ticker.lower() in url for ticker in tickers)
        
        # Check if source title contains keywords that appear in our news content
        title_words = set(title.split())
        news_words = set(news_lower.split())
        
        # Look for significant word overlap or ticker mentions
        word_overlap = len(title_words.intersection(news_words)) > 2
        
        if ticker_mentioned or word_overlap:
            relevant_sources.append(source)
    
    return relevant_sources

def fetch_news_batches(all_tickers, batch_size=5, recency="day", domains=VN_SOURCES):
    """Fetch news in smaller batches to reduce hallucination and improve accuracy."""
    all_summaries = []
    all_sources = []
    all_found_tickers = []
    
    # Split tickers into batches
    batches = [all_tickers[i:i + batch_size] for i in range(0, len(all_tickers), batch_size)]
    
    for i, batch in enumerate(batches):
        # Silently process batch without showing progress
        summary, sources, found_tickers = fetch_news(batch, recency=recency, domains=domains)
        
        if summary and summary != "Không có tin tức mới trong khoảng thời gian được yêu cầu.":
            all_summaries.append(summary)
        
        if sources:
            all_sources.extend(sources)
            
        if found_tickers:
            all_found_tickers.extend([t for t in found_tickers if t not in all_found_tickers])
    
    # Merge all summaries
    merged_summary = '\n\n'.join(all_summaries) if all_summaries else None
    
    return merged_summary, all_sources, all_found_tickers

def fetch_news_with_fallback(tickers, batch_size=5, domains=VN_SOURCES):
    """Fetch news with fallback from 'day' to 'week' if no results found."""
    
    # Try 'day' first - silently
    summary, sources, found_tickers = fetch_news_batches(tickers, batch_size, recency="day", domains=domains)
    
    if summary and len(summary.strip()) > 50:  # Has meaningful content
        return summary, sources, found_tickers, "day"
    
    # Fallback to 'week' if day search was empty or minimal - silently
    summary, sources, found_tickers = fetch_news_batches(tickers, batch_size, recency="week", domains=domains)
    
    return summary, sources, found_tickers, "week"

def extract_urls_from_text(text):
    """Extract URLs from text content."""
    import re
    url_pattern = r'https?://[^\s\)]+|www\.[^\s\)]+'
    urls = re.findall(url_pattern, text)
    return urls

def validate_and_display_news(summary, tickers, sources):
    """Validate news content and display only factual information with proper formatting."""
    if not summary:
        return False, []
    
    lines = summary.split('\n')
    valid_lines = []
    found_any_ticker = False
    displayed_urls = []
    
    # Patterns that indicate hallucination, non-factual content, or off-topic news
    hallucination_patterns = [
        "dự kiến",
        "có thể", 
        "dường như",
        "theo nguồn tin",
        "được cho là",
        "tin đồn",
        "chưa được xác nhận",
        "theo thông tin không chính thức",
        "không có tin tức",
        "không tìm thấy",
    ]
    
    # Trusted URL domains for validation
    trusted_domains = ["cafef.vn", "vnexpress.net", "vneconomy.vn", "fireant.vn"]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line contains ticker
        has_ticker = any(ticker.upper() in line.upper() for ticker in tickers)
        
        # Check for hallucination patterns
        has_hallucination = any(pattern in line.lower() for pattern in hallucination_patterns)
        
        # Check if line has trusted URL (stricter validation)
        has_trusted_url = any(domain in line.lower() for domain in trusted_domains)
        
        # Only include lines that:
        # 1. Contain a ticker
        # 2. Don't have hallucination patterns
        # 3. Have valid news indicators OR start with ticker formatting
        # 4. For stricter validation: contain trusted URLs
        if has_ticker and not has_hallucination :
            # For lines without URLs, be more lenient if they have strong factual indicators
            if has_trusted_url:
                valid_lines.append(line)
                found_any_ticker = True
                st.write(line)
                
                # Extract URLs from the displayed line
                line_urls = extract_urls_from_text(line)
                displayed_urls.extend(line_urls)
    
    if not found_any_ticker:
        st.success("✅ You're all caught up! No new verified news for your tickers.")
        return False, []
    
    return True, displayed_urls

def filter_sources_by_displayed_urls(sources, displayed_urls):
    """Filter sources to only include ones that were actually referenced in displayed content."""
    if not displayed_urls or not sources:
        return []
    
    relevant_sources = []
    
    for source in sources:
        source_url = source.get("url", "")
        # Check if this source URL matches any URL that was actually displayed
        for displayed_url in displayed_urls:
            if displayed_url in source_url or source_url in displayed_url:
                relevant_sources.append(source)
                break
    
    return relevant_sources

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
if 'used_recency' not in st.session_state:
    st.session_state.used_recency = 'day'
if 'last_news_date' not in st.session_state:
    st.session_state.last_news_date = None

# Brokerage & Banking News Alert (moved to top)
st.subheader("🏦 Brokerage & Banking News Alert")

# Manual refresh button only
col1, col2 = st.columns([1, 4])
with col1:
    load_banking_news = st.button("📰 Load Latest News")

if load_banking_news:
    with st.spinner("Fetching banking & brokerage news..."):
        # Use new batch approach with fallback
        predefined_summary, predefined_sources, found_tickers, used_recency = fetch_news_with_fallback(
            ALL_PREDEFINED_TICKERS, 
            batch_size=5
        )
    
    # Store in session state
    st.session_state.banking_summary = predefined_summary
    st.session_state.banking_sources = predefined_sources
    st.session_state.banking_found_tickers = found_tickers
    st.session_state.banking_news_loaded = True
    st.session_state.used_recency = used_recency

# Always display news if available
if st.session_state.banking_summary:
    with st.container():
        found_tickers_str = ", ".join(st.session_state.banking_found_tickers) if st.session_state.banking_found_tickers else "không xác định"
        recency_text = "last 2 days" if st.session_state.get('used_recency', 'day') == 'day' else "last 2 weeks"
        st.success(f"✅ Latest news for brokerage and banking stocks from {recency_text} (Found: {found_tickers_str}):")
        
        # Use validation function to display only factual content
        has_valid_news, displayed_urls = validate_and_display_news(
            st.session_state.banking_summary, 
            st.session_state.banking_found_tickers,
            st.session_state.banking_sources
        )
        
        if st.session_state.banking_sources:
            # Filter sources to only show ones that were actually referenced in displayed content
            relevant_sources = filter_sources_by_displayed_urls(
                st.session_state.banking_sources,
                displayed_urls
            )
            
            if relevant_sources:
                with st.expander("📋 Detailed Sources"):
                    for s in relevant_sources:
                        title = s.get("title", "(no title)")
                        url = s.get("url", "#")
                        date = s.get("date", "")
                        st.markdown(f"- [{title}]({url}) ({date})")
            else:
                with st.expander("📋 Detailed Sources"):
                    st.info("No relevant sources found for the displayed news.")
        
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

# Remove caching to ensure fresh results every time
# @st.cache_data(ttl=60*60)  # cache 1 hour - DISABLED for consistency
def get_fresh_news(tickers):
    return fetch_news(tickers)

# Custom ticker news search (only show when user enters tickers)
if tickers_list:
    # Always get fresh news for custom tickers using 2-week period
    summary, sources, found_tickers = fetch_news(tickers_list, recency="week")
    
    # Display validated news content
    if summary:
        has_valid_news, displayed_urls = validate_and_display_news(summary, tickers_list, sources)
        
        if not has_valid_news:
            st.warning("⚠️ No verified news found for the entered tickers in the past 2 weeks.")
    else:
        st.info("No news content returned from the API.")

    # Show filtered sources for custom tickers - only URLs that were actually displayed
    if sources and 'displayed_urls' in locals():
        relevant_sources = filter_sources_by_displayed_urls(sources, displayed_urls)
        
        if relevant_sources:
            st.subheader("🔗 News Sources")
            for s in relevant_sources:
                title = s.get("title", "(no title)")
                url = s.get("url", "#")
                date = s.get("date", "")
                st.markdown(f"- [{title}]({url}) ({date})")
        else:
            st.info("No relevant sources found for the displayed news content.")
    else:
        st.info("No news sources found for the entered tickers.")
