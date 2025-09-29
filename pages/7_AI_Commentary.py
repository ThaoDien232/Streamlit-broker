"""
AI-Powered Quarterly Business Performance Commentary
Generates intelligent analysis of broker financial performance using OpenAI
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Commentary",
    page_icon="🤖",
    layout="wide"
)

import pandas as pd
import toml
import os
from datetime import datetime

# Load theme from config.toml
theme_config = toml.load("utils/config.toml")
theme = theme_config["theme"]
primary_color = theme["primaryColor"]

# Import utilities
from utils.openai_commentary import (
    generate_commentary,
    get_available_tickers,
    get_available_quarters,
    get_broker_data
)

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_financial_data():
    """Load Combined_Financial_Data.csv with caching - same approach as Historical page"""
    try:
        df = pd.read_csv('sql/Combined_Financial_Data.csv', dtype={'TICKER': str}, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        return pd.DataFrame()

# Title and description
st.title("🤖 AI-Powered Business Commentary")
st.markdown("Generate intelligent quarterly analysis of broker financial performance using advanced AI")

# Manual refresh control
with st.sidebar:
    st.header("🔄 Data Controls")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

# Load data
df = load_financial_data()

if df.empty:
    st.error("❌ Financial data not found. Please ensure Combined_Financial_Data.csv exists in the sql/ directory.")
    st.stop()

# Get available options
available_tickers = get_available_tickers(df)
available_quarters = get_available_quarters(df)

# Check for cached commentary
cache_file = "sql/ai_commentary_cache.csv"
cache_exists = os.path.exists(cache_file)

if cache_exists:
    try:
        cache_df = pd.read_csv(cache_file)
        total_cached = len(cache_df)
        unique_tickers = cache_df['TICKER'].nunique()
        st.sidebar.success(f"📚 Cache: {total_cached} commentaries for {unique_tickers} brokers")
    except:
        cache_exists = False

# Main interface
st.subheader("Generate Quarterly Commentary")

# Input controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_ticker = st.selectbox(
        "Select Broker:",
        available_tickers,
        index=available_tickers.index('SSI') if 'SSI' in available_tickers else 0,
        help="Choose a broker to analyze"
    )

with col2:
    ticker_quarters = get_available_quarters(df, selected_ticker)
    selected_quarter = st.selectbox(
        "Select Quarter:",
        ticker_quarters,
        index=0,
        help="Choose the quarter to analyze or generate commentary for"
    )

with col3:
    model_choice = st.selectbox(
        "AI Model:",
        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4: Most capable (recommended)\ngpt-4-turbo: Faster\ngpt-3.5-turbo: Most economical"
    )

with col4:
    force_regenerate = st.checkbox(
        "Force Regenerate",
        value=False,
        help="Bypass cache and generate fresh analysis (costs API credits)"
    )

# Show broker information
if selected_ticker and selected_quarter:
    # Get financial data for preview
    broker_metrics = get_broker_data(selected_ticker, selected_quarter, df)

    if broker_metrics:
        # Format key metrics for display
        def format_number(value):
            if abs(value) >= 1e12:
                return f"{value/1e12:.1f}T VND"
            elif abs(value) >= 1e9:
                return f"{value/1e9:.1f}B VND"
            elif abs(value) >= 1e6:
                return f"{value/1e6:.1f}M VND"
            else:
                return f"{value:.0f} VND"

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Revenue", format_number(broker_metrics['revenue']))
        with col2:
            st.metric("Net Profit", format_number(broker_metrics['net_profit']))
        with col3:
            st.metric("ROA", f"{broker_metrics['roa']:.2f}%")
        with col4:
            st.metric("ROE", f"{broker_metrics['roe']:.2f}%")

        # Check if cached analysis exists
        if cache_exists and not force_regenerate:
            try:
                cached_analysis = cache_df[
                    (cache_df['TICKER'] == selected_ticker) &
                    (cache_df['QUARTER'] == selected_quarter)
                ]
                if not cached_analysis.empty:
                    latest_cached = cached_analysis.iloc[-1]
                    generated_date = pd.to_datetime(latest_cached['GENERATED_DATE']).strftime('%Y-%m-%d %H:%M')

                    st.success(f"📋 Cached analysis available (Generated: {generated_date})")

                    # Display cached commentary
                    st.subheader(f"AI Analysis: {selected_ticker} - {selected_quarter}")
                    st.markdown(latest_cached['COMMENTARY'])
                    st.caption(f"Model used: {latest_cached.get('MODEL', 'Unknown')}")
                else:
                    st.info(f"💡 No cached analysis found for {selected_ticker} - {selected_quarter}")
            except Exception as e:
                st.warning(f"Could not check cache: {e}")
    else:
        st.error(f"No financial data found for {selected_ticker} in {selected_quarter}")

# Generation controls
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    generate_button = st.button("🚀 Generate Analysis", type="primary")

with col2:
    if cache_exists:
        view_cache_button = st.button("📚 View All Cached")
    else:
        view_cache_button = False

# Generate commentary
if generate_button and selected_ticker and selected_quarter:
    if not broker_metrics:
        st.error(f"No financial data available for {selected_ticker} in {selected_quarter}")
    else:
        with st.spinner(f"🤖 Generating AI commentary for {selected_ticker} - {selected_quarter}..."):
            try:
                commentary = generate_commentary(
                    ticker=selected_ticker,
                    year_quarter=selected_quarter,
                    df=df,
                    model=model_choice,
                    force_regenerate=force_regenerate
                )

                if commentary.startswith("Error"):
                    st.error(commentary)
                    st.info("💡 **Tips:**\n- Check your OpenAI API key in .env file\n- Ensure you have API credits\n- Try a different model")
                else:
                    st.success("✅ Analysis generated successfully!")

                    # Display the generated commentary
                    st.subheader(f"AI Analysis: {selected_ticker} - {selected_quarter}")
                    st.markdown(commentary)
                    st.caption(f"Generated with {model_choice} on {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            except Exception as e:
                st.error(f"Error generating commentary: {e}")
                st.info("💡 **Common issues:**\n- Missing OpenAI API key\n- Invalid API key\n- Insufficient API credits\n- Network connectivity")

# View cached commentaries
if view_cache_button and cache_exists:
    st.subheader("📚 Cached AI Commentaries")

    try:
        cache_df = pd.read_csv(cache_file)
        cache_df['GENERATED_DATE'] = pd.to_datetime(cache_df['GENERATED_DATE']).dt.strftime('%Y-%m-%d %H:%M')

        # Display summary
        st.write(f"**Total cached analyses:** {len(cache_df)}")

        # Show recent analyses
        display_df = cache_df[['TICKER', 'QUARTER', 'GENERATED_DATE', 'MODEL']].sort_values('GENERATED_DATE', ascending=False)
        st.dataframe(display_df, use_container_width=True)

        # Allow selection and viewing of specific cached commentary
        if len(cache_df) > 0:
            st.subheader("View Specific Commentary")

            selected_cache = st.selectbox(
                "Select cached analysis:",
                options=range(len(cache_df)),
                format_func=lambda x: f"{cache_df.iloc[x]['TICKER']} - {cache_df.iloc[x]['QUARTER']} ({cache_df.iloc[x]['GENERATED_DATE']})"
            )

            if st.button("Show Selected Commentary"):
                selected_row = cache_df.iloc[selected_cache]
                st.subheader(f"Analysis: {selected_row['TICKER']} - {selected_row['QUARTER']}")
                st.markdown(selected_row['COMMENTARY'])
                st.caption(f"Generated: {selected_row['GENERATED_DATE']} | Model: {selected_row['MODEL']}")

    except Exception as e:
        st.error(f"Error loading cached commentaries: {e}")

# Setup instructions
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 Setup")
    with st.expander("Setup Instructions"):
        st.markdown("""
        **Required:**
        1. Get OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
        2. Add to `.env` file:
           ```
           OPENAI_API_KEY="your-api-key-here"
           ```
        3. Install OpenAI package:
           ```
           pip install openai
           ```

        **Features:**
        - ✅ Automatic caching to save API costs
        - ✅ Multiple AI models available
        - ✅ Comparative analysis included
        - ✅ Vietnamese brokerage market context
        """)

st.markdown("---")
st.caption("💡 **Tip:** Generated commentaries are automatically cached to save API costs. Use 'Force Regenerate' only when you need fresh analysis.")