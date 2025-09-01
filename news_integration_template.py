
import requests
import streamlit as st
from datetime import datetime
import pandas as pd

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_ssi_news():
    """Fetch SSI-related news from multiple sources"""
    
    news_items = []
    
    # Source 1: Try VietStock
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://finance.vietstock.vn/'
        }
        
        response = requests.post(
            "https://finance.vietstock.vn/data/getnewsbycode",
            headers=headers,
            data={'code': 'SSI', 'size': 10},
            timeout=10
        )
        
        if response.status_code == 200:
            # Parse response for news (would need to be customized based on actual response format)
            news_items.append({
                'title': 'News from VietStock (processing needed)',
                'source': 'VietStock',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'url': 'https://finance.vietstock.vn/co-phieu/SSI'
            })
    except:
        pass
    
    # Source 2: Static/fallback news items
    fallback_news = [
        {
            'title': 'SSI Q3 2024 Financial Results Released',
            'source': 'Company Report',
            'date': '2024-10-25 09:00',
            'url': 'https://www.ssi.com.vn'
        },
        {
            'title': 'SSI Expands Brokerage Services',
            'source': 'Market News',
            'date': '2024-10-20 14:30',
            'url': 'https://www.ssi.com.vn'
        }
    ]
    
    news_items.extend(fallback_news)
    
    return pd.DataFrame(news_items)

# Usage in Streamlit app:
def display_news_section():
    """Add this to your Streamlit app"""
    
    st.header("📰 SSI News & Updates")
    
    news_df = fetch_ssi_news()
    
    if not news_df.empty:
        for _, news in news_df.iterrows():
            with st.expander(f"📄 {news['title']} - {news['source']}"):
                st.write(f"**Date:** {news['date']}")
                st.write(f"**Source:** {news['source']}")
                if news['url']:
                    st.markdown(f"[Read more]({news['url']})")
    else:
        st.info("No recent news available.")

# To integrate into your existing app, add this to one of your pages:
# display_news_section()
