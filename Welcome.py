import streamlit as st

st.set_page_config(
    page_title="Broker Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar navigation - applies to all pages
FONT_SIZE = "18px"        # Font size - change as needed

st.markdown(f"""
<style>
/* Increase font size for sidebar navigation links */
[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li a {{
    font-size: {FONT_SIZE} !important;
    font-weight: 500 !important;
}}

/* Alternative selectors for different Streamlit versions */
.css-1d391kg .css-wjbhl0 {{
    font-size: {FONT_SIZE} !important;
    font-weight: 500 !important;
}}

.css-1d391kg a {{
    font-size: {FONT_SIZE} !important;
    font-weight: 500 !important;
}}

[data-testid="stSidebar"] .css-1d391kg > div {{
    font-size: {FONT_SIZE} !important;
    font-weight: 500 !important;
}}

/* Ensure font size applies to all navigation elements */
[data-testid="stSidebar"] * {{
    font-size: {FONT_SIZE} !important;
}}
</style>
""", unsafe_allow_html=True)

st.title("📈 Broker Analysis Platform")
st.markdown("---")

st.markdown("""
## Welcome to the Broker Analysis Platform

This platform provides comprehensive analysis tools for Vietnamese securities brokers. Use the sidebar to navigate between different analysis modules:

### 📊 Available Pages:
- **Prop Book Dashboard** - View and analyze proprietary trading positions
- **Charts** - Technical analysis and charting tools  
- **Forecast** - Financial forecasting and modeling

### 🚀 Getting Started:
1. Select a page from the sidebar
2. Each page has its own set of tools and filters
3. Data is automatically cached for better performance

### 📝 Features:
- Real-time price integration via TCBS API
- Interactive filtering and data export
- Multi-broker comparative analysis
- Profit/loss calculations and forecasting

---
*Select a page from the sidebar to begin your analysis*
""")

# Add some key metrics or summary information
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**Prop Book Dashboard**\n\nAnalyze broker proprietary trading positions with real-time P&L calculations")

with col2:
    st.info("**Charts & Analysis**\n\nTechnical analysis tools and visualization capabilities")

with col3:
    st.info("**Forecasting Models**\n\nFinancial modeling and prediction tools for broker performance")