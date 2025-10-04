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
    """Load Combined_Financial_Data.csv with caching"""
    try:
        df = pd.read_csv('sql/Combined_Financial_Data.csv', dtype={'TICKER': str}, low_memory=False)
        return df
    except Exception as e:
        st.error(f"Error loading financial data: {e}")
        return pd.DataFrame()

def filter_ticker_data(df, ticker):
    """Step 1: Filter data for specific ticker and return historical table"""
    ticker_data = df[df['TICKER'] == ticker].copy()

    # Pivot to get quarters as columns and metrics as rows
    pivot_data = ticker_data.pivot_table(
        index=['KEYCODE_NAME'],
        columns='QUARTER_LABEL',
        values='VALUE',
        aggfunc='first'
    ).reset_index()

    return ticker_data, pivot_data

def calculate_financial_metrics(ticker_data, selected_quarter, ticker):
    """Step 2: Calculate YoY, QoQ growth rates and financial ratios for selected quarter"""

    # Key financial metrics to extract using CALC statement type and METRIC_CODE (like Historical page)
    key_metrics = {
        'Net Brokerage Income': 'NET_BROKERAGE_INCOME',
        'Margin Income': 'MARGIN_LENDING_INCOME',
        'Investment Income': 'NET_INVESTMENT_INCOME',
        'PBT': 'PBT',
        'Margin Balance': 'MARGIN_BALANCE',
        'ROE': 'ROE'  # Use existing ROE calculation from CSV
    }

    # Get data for current quarter and comparison periods
    current_data = ticker_data[ticker_data['QUARTER_LABEL'] == selected_quarter]

    # Calculate metrics using CALC statement type
    metrics_dict = {}

    # First, extract year and quarter from the selected quarter
    # Parse quarter format like "1Q24", "2Q24", etc.
    if len(selected_quarter) >= 3:
        quarter_num = int(selected_quarter[0])  # Extract quarter number
        year_str = selected_quarter[-2:]  # Extract year part
        if year_str.isdigit():
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        else:
            # Fallback: try to find the year from the data
            year_data = current_data['YEARREPORT'].dropna()
            year = int(year_data.iloc[0]) if len(year_data) > 0 else 2024
    else:
        year = 2024
        quarter_num = 1

    # Use only the key metrics
    all_metrics = key_metrics

    for metric_name, metric_code in all_metrics.items():
        current_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, metric_code)
        metrics_dict[f'{metric_name}_Current'] = current_value

    # Get previous quarter for QoQ calculation (e.g., 1Q24 vs 4Q23)
    quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q) and q != ''])
    if selected_quarter in quarters:
        current_idx = quarters.index(selected_quarter)
        prev_quarter = quarters[current_idx - 1] if current_idx > 0 else None

        if prev_quarter:
            # Parse previous quarter to get year and quarter number
            if len(prev_quarter) >= 3:
                prev_quarter_num = int(prev_quarter[0])
                prev_year_str = prev_quarter[-2:]
                if prev_year_str.isdigit():
                    prev_year = 2000 + int(prev_year_str) if int(prev_year_str) < 50 else 1900 + int(prev_year_str)
                else:
                    prev_year = year - 1  # Fallback
            else:
                prev_year = year
                prev_quarter_num = quarter_num - 1

            for metric_name, metric_code in all_metrics.items():
                prev_value = get_calc_metric_value(ticker_data, ticker, prev_year, prev_quarter_num, metric_code)
                metrics_dict[f'{metric_name}_Previous'] = prev_value

                # Calculate QoQ growth
                current_val = metrics_dict.get(f'{metric_name}_Current')
                if current_val and prev_value and prev_value != 0:
                    qoq_growth = ((current_val - prev_value) / abs(prev_value)) * 100
                    metrics_dict[f'{metric_name}_QoQ_Growth'] = qoq_growth

    # Get same quarter last year for YoY calculation (e.g., 1Q24 vs 1Q23)
    yoy_year = year - 1  # Same quarter, previous year

    for metric_name, metric_code in all_metrics.items():
        yoy_value = get_calc_metric_value(ticker_data, ticker, yoy_year, quarter_num, metric_code)
        metrics_dict[f'{metric_name}_YoY'] = yoy_value

        # Calculate YoY growth
        current_val = metrics_dict.get(f'{metric_name}_Current')
        if current_val and yoy_value and yoy_value != 0:
            yoy_growth = ((current_val - yoy_value) / abs(yoy_value)) * 100
            metrics_dict[f'{metric_name}_YoY_Growth'] = yoy_growth

    # ROE is already calculated and extracted from the CSV data above

    return pd.DataFrame([metrics_dict])

def create_analysis_table(ticker_data, calculated_metrics, selected_quarter):
    """Step 3: Combine historical data and calculated metrics for OpenAI analysis"""

    # Get key financial statement items for the selected quarter
    quarter_data = ticker_data[ticker_data['QUARTER_LABEL'] == selected_quarter]

    # Get dynamic column headers based on actual quarters
    quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q) and q != ''])

    # Find previous quarter and same quarter last year
    if selected_quarter in quarters:
        current_idx = quarters.index(selected_quarter)
        prev_quarter = quarters[current_idx - 1] if current_idx > 0 else "N/A"
    else:
        prev_quarter = "N/A"

    # Calculate same quarter last year
    try:
        quarter_num = int(selected_quarter[0])
        year_str = selected_quarter[-2:]
        year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        yoy_quarter = f"{quarter_num}Q{(year-1) % 100:02d}"
    except:
        yoy_quarter = "N/A"

    # Create summary table with dynamic headers
    analysis_data = {
        'Metric': [],
        f'Current Value ({selected_quarter})': [],
        f'Previous Quarter ({prev_quarter})': [],
        'QoQ Growth %': [],
        f'Same Quarter Last Year ({yoy_quarter})': [],
        'YoY Growth %': []
    }

    if not calculated_metrics.empty:
        metrics_row = calculated_metrics.iloc[0]

        # Show only the requested metrics with friendly names
        display_metrics = ['Net Brokerage Income', 'Margin Income', 'Investment Income', 'PBT', 'Margin Balance', 'ROE']

        for metric in display_metrics:
            analysis_data['Metric'].append(metric)
            analysis_data[f'Current Value ({selected_quarter})'].append(metrics_row.get(f'{metric}_Current', 'N/A'))

            # Don't calculate growth for ROE and ROA, but still show historical data
            if metric in ['ROE', 'ROA']:
                analysis_data[f'Previous Quarter ({prev_quarter})'].append(metrics_row.get(f'{metric}_Previous', 'N/A'))
                analysis_data['QoQ Growth %'].append('N/A')
                analysis_data[f'Same Quarter Last Year ({yoy_quarter})'].append(metrics_row.get(f'{metric}_YoY', 'N/A'))
                analysis_data['YoY Growth %'].append('N/A')
            else:
                analysis_data[f'Previous Quarter ({prev_quarter})'].append(metrics_row.get(f'{metric}_Previous', 'N/A'))
                analysis_data['QoQ Growth %'].append(metrics_row.get(f'{metric}_QoQ_Growth', 'N/A'))
                analysis_data[f'Same Quarter Last Year ({yoy_quarter})'].append(metrics_row.get(f'{metric}_YoY', 'N/A'))
                analysis_data['YoY Growth %'].append(metrics_row.get(f'{metric}_YoY_Growth', 'N/A'))

    return pd.DataFrame(analysis_data)

def get_calc_metric_value(df, ticker, year, quarter, metric_code):
    """Get a specific calculated metric value from CALC statement type"""
    result = df[
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter) &
        (df['STATEMENT_TYPE'] == 'CALC') &
        (df['METRIC_CODE'] == metric_code)
    ]

    if len(result) > 0:
        value = result.iloc[0]['VALUE']

        # Annualize quarterly ROE and ROA (multiply by 4 for quarters 1-4, not for annual which is 5)
        if metric_code in ['ROE', 'ROA'] and quarter in [1, 2, 3, 4]:
            value = value * 4

        return value
    return 0

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

def sort_quarters_chronologically(quarters):
    """Sort quarters in chronological order (1Q19, 2Q19, 3Q19, 4Q19, 1Q20, etc.)"""
    def quarter_key(quarter):
        if pd.isna(quarter) or quarter == 'Annual':
            return (9999, 0)  # Put invalid quarters at the end
        try:
            # Parse quarters like "1Q19", "2Q20", etc.
            quarter_num = int(quarter[0])
            year_str = quarter[-2:]
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
            return (year, quarter_num)
        except (ValueError, IndexError):
            return (9999, 0)  # Invalid format goes to end

    return sorted(quarters, key=quarter_key)

# Get available options from the CSV data
available_tickers = sorted([t for t in df['TICKER'].unique() if pd.notna(t)], key=str)
available_quarters = sort_quarters_chronologically([q for q in df['QUARTER_LABEL'].unique() if pd.notna(q) and q != 'Annual'])

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
    # Filter quarters available for the selected ticker
    ticker_df = df[df['TICKER'] == selected_ticker]
    ticker_quarters = sort_quarters_chronologically([q for q in ticker_df['QUARTER_LABEL'].unique() if pd.notna(q) and q != 'Annual'])

    if ticker_quarters:
        selected_quarter = st.selectbox(
            "Select Quarter:",
            ticker_quarters,
            index=0,
            help="Choose the quarter to analyze or generate commentary for"
        )
    else:
        st.error(f"No quarterly data found for {selected_ticker}")
        selected_quarter = None

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

# Show broker information using 3-step approach
if selected_ticker and selected_quarter:
    try:
        # Step 1: Filter ticker data
        ticker_data, pivot_data = filter_ticker_data(df, selected_ticker)

        if ticker_data.empty:
            st.error(f"No financial data found for {selected_ticker}")
            st.stop()

        # Step 2: Calculate financial metrics and growth rates
        calculated_metrics = calculate_financial_metrics(ticker_data, selected_quarter, selected_ticker)

        # Step 3: Create analysis table for OpenAI
        analysis_table = create_analysis_table(ticker_data, calculated_metrics, selected_quarter)

        # Display the data processing results
        st.subheader(f"📊 Financial Analysis: {selected_ticker} - {selected_quarter}")

        # Show calculated metrics
        if not calculated_metrics.empty and not analysis_table.empty:
            # Format numbers for display
            def format_number(value, metric_name):
                if pd.isna(value) or value == 'N/A':
                    return "N/A"
                try:
                    value = float(value)
                    # ROE and ROA should be displayed as percentages (multiply by 100 since CSV has decimal values)
                    if metric_name in ['ROE', 'ROA']:
                        return f"{value * 100:.2f}%"
                    # Other financial metrics in billions VND with thousand separators
                    elif abs(value) >= 1e9:
                        return f"{value/1e9:,.1f}B VND"
                    elif abs(value) >= 1e6:
                        return f"{value/1e6:,.1f}M VND"
                    else:
                        return f"{value:,.0f} VND"
                except:
                    return str(value)

            def format_growth(value):
                if pd.isna(value) or value == 'N/A':
                    return "N/A"
                try:
                    value = float(value)
                    return f"{value:+.1f}%"
                except:
                    return str(value)

            # Display analysis table
            st.write("**Financial Metrics Summary:**")
            display_table = analysis_table.copy()

            # Format the values based on metric type
            for idx, row in display_table.iterrows():
                metric_name = row['Metric']

                # Format current value column
                current_col = [col for col in display_table.columns if col.startswith('Current Value')][0]
                if row[current_col] != 'N/A':
                    display_table.at[idx, current_col] = format_number(row[current_col], metric_name)

                # Format previous quarter column
                prev_col = [col for col in display_table.columns if col.startswith('Previous Quarter')][0]
                if row[prev_col] != 'N/A':
                    display_table.at[idx, prev_col] = format_number(row[prev_col], metric_name)

                # Format same quarter last year column
                yoy_col = [col for col in display_table.columns if col.startswith('Same Quarter Last Year')][0]
                if row[yoy_col] != 'N/A':
                    display_table.at[idx, yoy_col] = format_number(row[yoy_col], metric_name)

                # Format growth columns
                qoq_col = 'QoQ Growth %'
                if row[qoq_col] != 'N/A':
                    display_table.at[idx, qoq_col] = format_growth(row[qoq_col])

                yoy_growth_col = 'YoY Growth %'
                if row[yoy_growth_col] != 'N/A':
                    display_table.at[idx, yoy_growth_col] = format_growth(row[yoy_growth_col])

            st.dataframe(display_table, use_container_width=True)

            # Show raw metrics for debugging (expandable)
            with st.expander("🔍 Detailed Calculation Data"):
                st.write("**Raw Historical Data (last 10 quarters):**")
                recent_quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q)])[-10:]
                recent_data = ticker_data[ticker_data['QUARTER_LABEL'].isin(recent_quarters)]
                st.dataframe(recent_data[['QUARTER_LABEL', 'KEYCODE_NAME', 'VALUE']].head(20))

                # Special debug for ROE if SSI is selected
                if selected_ticker == 'SSI':
                    st.write("**ROE Debug for SSI (1Q24 to 1Q25):**")
                    roe_debug_quarters = ['1Q24', '2Q24', '3Q24', '4Q24', '1Q25']
                    roe_debug_data = []

                    for quarter in roe_debug_quarters:
                        # Parse quarter to get year and quarter number
                        try:
                            quarter_num = int(quarter[0])
                            year_str = quarter[-2:]
                            year = 2000 + int(year_str)

                            roe_value = get_calc_metric_value(ticker_data, selected_ticker, year, quarter_num, 'ROE')
                            roe_debug_data.append({
                                'Quarter': quarter,
                                'Year': year,
                                'Quarter_Num': quarter_num,
                                'ROE_Value': roe_value
                            })
                        except:
                            roe_debug_data.append({
                                'Quarter': quarter,
                                'Year': 'Error',
                                'Quarter_Num': 'Error',
                                'ROE_Value': 'Error'
                            })

                    st.dataframe(pd.DataFrame(roe_debug_data))

                    # Also show raw ROE data from the dataset
                    st.write("**Raw ROE data from CSV:**")
                    roe_raw = ticker_data[
                        (ticker_data['STATEMENT_TYPE'] == 'CALC') &
                        (ticker_data['METRIC_CODE'] == 'ROE') &
                        (ticker_data['QUARTER_LABEL'].isin(roe_debug_quarters))
                    ][['QUARTER_LABEL', 'YEARREPORT', 'LENGTHREPORT', 'VALUE']].sort_values('QUARTER_LABEL')
                    st.dataframe(roe_raw)

                st.write("**Calculated Metrics:**")
                st.dataframe(calculated_metrics)

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
            st.warning(f"Could not calculate metrics for {selected_ticker} in {selected_quarter}")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.stop()

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
    try:
        # Ensure we have the analysis data
        ticker_data, pivot_data = filter_ticker_data(df, selected_ticker)
        calculated_metrics = calculate_financial_metrics(ticker_data, selected_quarter, selected_ticker)
        analysis_table = create_analysis_table(ticker_data, calculated_metrics, selected_quarter)

        if analysis_table.empty:
            st.error(f"No financial data available for {selected_ticker} in {selected_quarter}")
        else:
            with st.spinner(f"🤖 Generating AI commentary for {selected_ticker} - {selected_quarter}..."):
                try:
                    # Convert analysis table to string for OpenAI
                    analysis_text = analysis_table.to_string(index=False)

                    commentary = generate_commentary(
                        ticker=selected_ticker,
                        year_quarter=selected_quarter,
                        df=analysis_table,  # Pass the processed analysis table instead of raw data
                        model=model_choice,
                        force_regenerate=force_regenerate
                    )

                    if commentary.startswith("Error"):
                        st.error(commentary)
                        st.info("💡 **Tips:**\n- Check your OpenAI API key in .streamlit/secrets.toml file\n- Ensure you have API credits\n- Try a different model")
                    else:
                        st.success("✅ Analysis generated successfully!")

                        # Display the generated commentary
                        st.subheader(f"AI Analysis: {selected_ticker} - {selected_quarter}")
                        st.markdown(commentary)
                        st.caption(f"Generated with {model_choice} on {datetime.now().strftime('%Y-%m-%d %H:%M')}")

                except Exception as e:
                    st.error(f"Error generating commentary: {e}")
                    st.info("💡 **Common issues:**\n- Missing OpenAI API key\n- Invalid API key\n- Insufficient API credits\n- Network connectivity")

    except Exception as e:
        st.error(f"Error preparing data for analysis: {e}")

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
    st.subheader("⚙️ Metric Configuration")
    with st.expander("Add More Metrics"):
        st.markdown("""
        **Currently Displayed:**
        - Net Brokerage Income (QoQ, YoY growth)
        - Margin Income (QoQ, YoY growth)
        - Investment Income (QoQ, YoY growth)
        - PBT (QoQ, YoY growth)
        - Margin Balance (QoQ, YoY growth)
        - ROE (no growth %, shows historical values) - annualized for quarterly data

        **Available for Addition:**
        - ROA (Return on Assets)
        - NET_TRADING_INCOME
        - FEE_INCOME
        - INTEREST_INCOME
        - SGA
        - INTEREST_EXPENSE
        - NPAT
        - BORROWING_BALANCE

        To add more metrics, modify the `key_metrics` dictionary
        in the `calculate_financial_metrics()` function.
        """)

    st.markdown("---")
    st.subheader("🔧 Setup")
    with st.expander("Setup Instructions"):
        st.markdown("""
        **Required:**
        1. Get OpenAI API key from [platform.openai.com](https://platform.openai.com/api-keys)
        2. Add to `.streamlit/secrets.toml` file:
           ```
           [openai]
           api_key = "your-api-key-here"
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
        - ✅ 3-step data processing approach
        """)

st.markdown("---")
st.caption("💡 **Tip:** Generated commentaries are automatically cached to save API costs. Use 'Force Regenerate' only when you need fresh analysis.")