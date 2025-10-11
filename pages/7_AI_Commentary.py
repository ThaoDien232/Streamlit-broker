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
import requests
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
def load_ticker_data(ticker: str, quarter_label: str):
    """Load brokerage financial data for specific ticker and quarter (with lookback)"""
    try:
        from utils.brokerage_data import load_ticker_quarter_data
        # Load data for this ticker with 6 quarters lookback
        df = load_ticker_quarter_data(ticker=ticker, quarter_label=quarter_label, lookback_quarters=6)
        return df
    except Exception as e:
        st.error(f"Error loading financial data from database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_market_liquidity_data():
    """Load and calculate quarterly market liquidity from database"""
    try:
        from utils.market_index_data import load_market_liquidity_data as load_db_liquidity
        quarterly_liquidity = load_db_liquidity(start_year=2017)
        return quarterly_liquidity
    except Exception as e:
        st.warning(f"Could not load market liquidity data: {e}")
        return pd.DataFrame()

def filter_ticker_data(df, ticker):
    """Step 1: Filter data for specific ticker and return historical table (data already filtered by load_ticker_data)"""
    ticker_data = df.copy()  # Data already filtered in query

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
        'PBT': 'NET ACCOUNTING PROFIT/(LOSS) BEFORE TAX',
        'NPAT': 'NET PROFIT/(LOSS) AFTER TAX',
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
    """Step 3: Combine historical data and calculated metrics for OpenAI analysis - Last 6 Quarters"""

    # Get all quarters sorted chronologically
    quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q) and q != ''])

    # Find the index of selected quarter and get last 6 quarters
    if selected_quarter not in quarters:
        return pd.DataFrame()

    current_idx = quarters.index(selected_quarter)
    # Get last 6 quarters including current (or fewer if not available)
    last_6_quarters = quarters[max(0, current_idx - 5):current_idx + 1]

    if len(last_6_quarters) == 0:
        return pd.DataFrame()

    # Get ticker from ticker_data
    if ticker_data.empty or 'TICKER' not in ticker_data.columns:
        return pd.DataFrame()

    ticker = ticker_data['TICKER'].iloc[0]

    # Load market liquidity data
    market_liquidity_df = load_market_liquidity_data()

    # Display metrics we want to show
    display_metrics = ['Net Brokerage Income', 'Market Liquidity (Avg Daily)', 'Margin Income', 'Investment Income', 'PBT', 'NPAT', 'Margin Balance', 'Margin/Equity %', 'ROE']

    # Create table structure: Metric as rows, quarters as columns
    analysis_data = {'Metric': display_metrics}

    # For each of the last 6 quarters, get the metric values
    for quarter in last_6_quarters:
        # Parse quarter to get year and quarter_num
        try:
            quarter_num = int(quarter[0])
            year_str = quarter[-2:]
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        except:
            continue

        # Get metrics for this quarter
        quarter_values = []
        margin_balance_value = None
        total_equity_value = None

        for metric_name in display_metrics:
            if metric_name == 'Market Liquidity (Avg Daily)':
                # Get market liquidity for this quarter
                if not market_liquidity_df.empty:
                    liquidity_row = market_liquidity_df[
                        (market_liquidity_df['Year'] == year) &
                        (market_liquidity_df['Quarter'] == quarter_num)
                    ]
                    if not liquidity_row.empty:
                        quarter_values.append(liquidity_row.iloc[0]['Avg Daily Turnover (B VND)'])
                    else:
                        quarter_values.append(0)
                else:
                    quarter_values.append(0)
                continue

            if metric_name == 'Margin/Equity %':
                # Calculate margin/equity ratio
                if margin_balance_value is not None and total_equity_value is not None and total_equity_value != 0:
                    ratio = (margin_balance_value / total_equity_value) * 100
                    quarter_values.append(ratio)
                else:
                    # Need to fetch if not already fetched
                    if margin_balance_value is None:
                        margin_balance_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'MARGIN_BALANCE')
                    if total_equity_value is None:
                        total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'TOTAL_EQUITY')

                    if total_equity_value and total_equity_value != 0:
                        ratio = (margin_balance_value / total_equity_value) * 100
                        quarter_values.append(ratio)
                    else:
                        quarter_values.append(0)
                continue

            metric_code = {
                'Net Brokerage Income': 'NET_BROKERAGE_INCOME',
                'Margin Income': 'MARGIN_LENDING_INCOME',
                'Investment Income': 'NET_INVESTMENT_INCOME',
                'PBT': 'NET ACCOUNTING PROFIT/(LOSS) BEFORE TAX',
                'NPAT': 'NET PROFIT/(LOSS) AFTER TAX',
                'Margin Balance': 'MARGIN_BALANCE',
                'ROE': 'ROE'
            }.get(metric_name)

            value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, metric_code)
            quarter_values.append(value)

            # Store for ratio calculation
            if metric_name == 'Margin Balance':
                margin_balance_value = value
            elif metric_name == 'ROE':
                # Also get Total Equity for the ratio
                total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'TOTAL_EQUITY')

        # Add this quarter's column
        analysis_data[quarter] = quarter_values

    # Create DataFrame
    df_analysis = pd.DataFrame(analysis_data)

    # Check if we have any quarter columns besides 'Metric'
    if len(df_analysis.columns) <= 1:
        return pd.DataFrame()

    # Now add growth columns for the most recent quarter (selected_quarter)
    # Add QoQ and YoY growth as additional columns after the last quarter
    if len(last_6_quarters) >= 2 and selected_quarter in df_analysis.columns:
        prev_quarter = last_6_quarters[-2]
        if prev_quarter in df_analysis.columns:
            qoq_growth = []
            for metric in display_metrics:
                if metric in ['ROE', 'ROA']:
                    qoq_growth.append('N/A')
                else:
                    try:
                        current_val = df_analysis[df_analysis['Metric'] == metric][selected_quarter].values[0]
                        prev_val = df_analysis[df_analysis['Metric'] == metric][prev_quarter].values[0]
                        if prev_val != 0 and prev_val != 'N/A' and current_val != 'N/A':
                            growth = ((current_val - prev_val) / abs(prev_val)) * 100
                            qoq_growth.append(growth)
                        else:
                            qoq_growth.append('N/A')
                    except (IndexError, KeyError):
                        qoq_growth.append('N/A')

            df_analysis['QoQ Growth %'] = qoq_growth

    # Add YoY growth if we have at least 5 quarters
    if len(last_6_quarters) >= 5:
        yoy_quarter = last_6_quarters[-5]  # 4 quarters ago
        if yoy_quarter in df_analysis.columns and selected_quarter in df_analysis.columns:
            yoy_growth = []
            for metric in display_metrics:
                if metric in ['ROE', 'ROA']:
                    yoy_growth.append('N/A')
                else:
                    try:
                        current_val = df_analysis[df_analysis['Metric'] == metric][selected_quarter].values[0]
                        yoy_val = df_analysis[df_analysis['Metric'] == metric][yoy_quarter].values[0]
                        if yoy_val != 0 and yoy_val != 'N/A' and current_val != 'N/A':
                            growth = ((current_val - yoy_val) / abs(yoy_val)) * 100
                            yoy_growth.append(growth)
                        else:
                            yoy_growth.append('N/A')
                    except (IndexError, KeyError):
                        yoy_growth.append('N/A')

            df_analysis['YoY Growth %'] = yoy_growth

    return df_analysis

def get_calc_metric_value(df, ticker, year, quarter, metric_code):
    """Get a specific calculated metric value from CALC statement type"""
    # For PBT and NPAT, use KEYCODE_NAME instead of METRIC_CODE
    if metric_code in ['NET ACCOUNTING PROFIT/(LOSS) BEFORE TAX', 'NET PROFIT/(LOSS) AFTER TAX']:
        result = df[
            (df['TICKER'] == ticker) &
            (df['YEARREPORT'] == year) &
            (df['LENGTHREPORT'] == quarter) &
            (df['KEYCODE_NAME'] == metric_code)
        ]
    else:
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

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_market_share(ticker, quarter_label):
    """Fetch market share for a specific broker and quarter from HSX API"""
    try:
        # Mapping from our ticker codes to HSX API brokerage codes
        ticker_to_brokerage_code = {
            'VCI': 'Vietcap',
            'HCM': 'HSC',
            'VND': 'VNDS',
            'FTS': 'FPTS'
        }

        # Get the brokerage code for API lookup, default to ticker if not in mapping
        api_ticker = ticker_to_brokerage_code.get(ticker, ticker)

        # Parse quarter (e.g., "1Q24" -> year=2024, quarter=1)
        quarter_num = int(quarter_label[0])
        year_str = quarter_label[-2:]
        year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)

        url = "https://api.hsx.vn/s/api/v1/1/brokeragemarketshare/top/ten"
        params = {
            'pageIndex': 1,
            'pageSize': 30,
            'year': year,
            'period': quarter_num,
            'dateType': 1
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        if data.get('success') and 'data' in data:
            brokerage_data = data['data'].get('brokerageStock', [])

            for item in brokerage_data:
                if item.get('shortenName', '') == api_ticker:
                    return {
                        'market_share': item.get('percentage', 0),
                        'rank': brokerage_data.index(item) + 1
                    }

        return {'market_share': 0, 'rank': None}

    except Exception as e:
        return {'market_share': 0, 'rank': None}

@st.cache_data(ttl=1800)
def get_top_prop_holdings(ticker, quarter_label):
    """Get top 5 proprietary trading holdings from Prop book.xlsx"""
    try:
        prop_df = pd.read_excel('sql/Prop book.xlsx')

        # Filter for the specific broker and quarter
        broker_data = prop_df[
            (prop_df['Broker'] == ticker) &
            (prop_df['Quarter'] == quarter_label)
        ]

        if broker_data.empty:
            return []

        # Exclude PBT and Other entries
        broker_data = broker_data[~broker_data['Ticker'].isin(['PBT', 'Other AFS', 'Other FVTPL', 'Others'])]

        # Calculate total value from both FVTPL and AFS
        broker_data['Total_Value'] = broker_data['FVTPL value'].fillna(0) + broker_data['AFS value'].fillna(0)

        # Determine the type (FVTPL or AFS) based on which has value
        def get_holding_type(row):
            if pd.notna(row['FVTPL value']) and row['FVTPL value'] > 0:
                return 'FVTPL'
            elif pd.notna(row['AFS value']) and row['AFS value'] > 0:
                return 'AFS'
            return 'Unknown'

        broker_data['Type'] = broker_data.apply(get_holding_type, axis=1)

        # Get top 5 holdings by total value
        top_holdings = broker_data.nlargest(5, 'Total_Value')[['Ticker', 'Total_Value', 'Type']].to_dict('records')

        return top_holdings

    except Exception as e:
        return []

def get_investment_composition(ticker_data, ticker, quarter_label):
    """Get investment book composition with sub-categories (Equities, Bonds, etc.) for selected quarter"""
    from utils.investment_book import get_investment_data

    # Parse quarter to get year and quarter number
    try:
        quarter_num = int(quarter_label[0])
        year_str = quarter_label[-2:]
        year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
    except:
        return pd.DataFrame()

    # Get investment data for this quarter
    investment_data = get_investment_data(ticker_data, ticker, year, quarter_num)

    if not investment_data:
        return pd.DataFrame()

    # Build detailed composition with sub-categories
    # FVTPL and AFS: use Market Value
    # HTM: use Cost (measured at amortized cost)
    composition = []
    total_value = 0
    category_totals = {}

    for category in ['FVTPL', 'AFS', 'HTM']:
        if category not in investment_data or not investment_data[category]:
            continue

        category_total = 0

        # Determine which valuation to use
        if category == 'HTM':
            valuation_dict = investment_data[category].get('Cost', {})
        else:
            valuation_dict = investment_data[category].get('Market Value', {})

        if not isinstance(valuation_dict, dict):
            continue

        # Group by sub-categories (instrument types)
        sub_category_map = {}
        for item_name, item_value in valuation_dict.items():
            if item_value > 0:
                # Determine sub-category from item name
                if 'Bond' in item_name or 'bond' in item_name:
                    sub_cat = 'Bonds'
                elif 'CD' in item_name and 'CDs' in item_name:
                    sub_cat = 'CDs'
                elif 'Term Deposit' in item_name:
                    sub_cat = 'Term Deposits'
                elif 'Deposit' in item_name or 'deposit' in item_name:
                    # Generic deposit - categorize based on category
                    if category == 'HTM':
                        sub_cat = 'Term Deposits'
                    else:
                        sub_cat = 'CDs'
                elif 'Money Market' in item_name or 'Monetary market' in item_name:
                    sub_cat = 'Money Market Instruments'
                elif 'Share' in item_name or 'Fund' in item_name or 'Equity' in item_name or 'Equit' in item_name:
                    # Further breakdown equities
                    if 'Listed' in item_name and 'Unlisted' not in item_name:
                        sub_cat = 'Listed Equities'
                    elif 'Unlisted' in item_name:
                        sub_cat = 'Unlisted Equities'
                    elif 'Fund' in item_name:
                        sub_cat = 'Fund Certificates'
                    else:
                        sub_cat = 'Equities'
                else:
                    sub_cat = 'Others'

                if sub_cat not in sub_category_map:
                    sub_category_map[sub_cat] = 0
                sub_category_map[sub_cat] += item_value
                category_total += item_value

        # Add category header
        if category_total > 0:
            composition.append({
                'Category': category,
                'Sub-Category': '',
                'Value': category_total,
                'is_header': True
            })
            category_totals[category] = category_total
            total_value += category_total

            # Add sub-categories
            for sub_cat, sub_value in sorted(sub_category_map.items()):
                composition.append({
                    'Category': '',
                    'Sub-Category': f'  {sub_cat}',
                    'Value': sub_value,
                    'is_header': False
                })

    if total_value == 0:
        return pd.DataFrame()

    # Calculate percentages and format
    for item in composition:
        item['Value (B VND)'] = f"{item['Value'] / 1_000_000_000:,.1f}"
        item['Composition %'] = f"{(item['Value'] / total_value * 100):.1f}%"
        del item['Value']
        del item['is_header']

    # Add total row
    composition.append({
        'Category': 'Total',
        'Sub-Category': '',
        'Value (B VND)': f"{total_value / 1_000_000_000:,.1f}",
        'Composition %': '100.0%'
    })

    # Combine Category and Sub-Category into single column for display
    df = pd.DataFrame(composition)
    df['Investment Type'] = df.apply(
        lambda row: row['Category'] if row['Category'] else row['Sub-Category'],
        axis=1
    )

    return df[['Investment Type', 'Value (B VND)', 'Composition %']]

def create_summary_tables(ticker, quarter_label, ticker_data):
    """Step 4: Create separate tables for market share and prop book data - Last 6 Quarters"""

    # Get all quarters sorted chronologically
    quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q) and q != ''])

    # Find the index of selected quarter and get last 6 quarters
    if quarter_label not in quarters:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    current_idx = quarters.index(quarter_label)
    # Get last 6 quarters including current (or fewer if not available)
    last_6_quarters = quarters[max(0, current_idx - 5):current_idx + 1]

    # Build market share table for last 6 quarters
    market_share_table = pd.DataFrame()
    market_data = {'Quarter': []}
    market_share_values = []
    market_rank_values = []

    for quarter in last_6_quarters:
        market_share_data = fetch_market_share(ticker, quarter)
        market_data['Quarter'].append(quarter)
        if market_share_data['market_share'] > 0:
            market_share_values.append(f"{market_share_data['market_share']:.2f}%")
            market_rank_values.append(f"#{market_share_data['rank']}" if market_share_data['rank'] else 'N/A')
        else:
            market_share_values.append('N/A')
            market_rank_values.append('N/A')

    if market_data['Quarter']:
        market_data['Market Share'] = market_share_values
        market_data['Rank'] = market_rank_values
        market_share_table = pd.DataFrame(market_data)

    # Build prop holdings table for last 6 quarters (top 5 holdings per quarter)
    # Get prop holdings for current quarter only (showing evolution across quarters would be too complex)
    prop_holdings = get_top_prop_holdings(ticker, quarter_label)

    prop_holdings_table = pd.DataFrame()
    if prop_holdings:
        holdings_data = {
            'Ticker': [],
            'Value (B VND)': [],
            'Type': []
        }
        for holding in prop_holdings:
            holdings_data['Ticker'].append(holding['Ticker'])
            holdings_data['Value (B VND)'].append(f"{holding['Total_Value']:.1f}")
            holdings_data['Type'].append(holding['Type'])
        prop_holdings_table = pd.DataFrame(holdings_data)

    # Build investment composition table for current quarter
    investment_composition_table = get_investment_composition(ticker_data, ticker, quarter_label)

    return market_share_table, prop_holdings_table, investment_composition_table

# Title and description
st.title("AI-Powered Commentary")

# Manual refresh control
with st.sidebar:
    st.header("🔄 Data Controls")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

# Load available tickers and quarters (lightweight queries)
from utils.brokerage_data import get_available_tickers, get_ticker_quarters_list

available_tickers = get_available_tickers()

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

# Check for cached commentary
cache_file = "sql/ai_commentary_cache.csv"
cache_exists = os.path.exists(cache_file)

if cache_exists:
    try:
        cache_df = pd.read_csv(cache_file)
        total_cached = len(cache_df)
        unique_tickers = cache_df['TICKER'].nunique()
        st.sidebar.success(f"Cache: {total_cached} commentaries for {unique_tickers} brokers")
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
    # Get quarters available for the selected ticker (lightweight query)
    ticker_quarters = get_ticker_quarters_list(selected_ticker, start_year=2017)

    if ticker_quarters:
        selected_quarter = st.selectbox(
            "Select Quarter:",
            ticker_quarters,
            index=0,  # Default to latest quarter (already sorted newest first)
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
        # Load data ONLY for selected ticker and quarter (with lookback)
        ticker_data = load_ticker_data(selected_ticker, selected_quarter)

        if ticker_data.empty:
            st.error(f"No financial data found for {selected_ticker} - {selected_quarter}")
            st.stop()

        # Step 1: Filter ticker data (already filtered, just format)
        ticker_data, pivot_data = filter_ticker_data(ticker_data, selected_ticker)

        # Step 2: Calculate financial metrics and growth rates
        calculated_metrics = calculate_financial_metrics(ticker_data, selected_quarter, selected_ticker)

        # Step 3: Create analysis table for OpenAI
        try:
            analysis_table = create_analysis_table(ticker_data, calculated_metrics, selected_quarter)
        except Exception as e:
            st.error(f"Error in create_analysis_table: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            analysis_table = pd.DataFrame()

        # Step 4: Create summary tables with market share, prop book, and investment composition data
        market_share_table, prop_holdings_table, investment_composition_table = create_summary_tables(selected_ticker, selected_quarter, ticker_data)

        # Display the data processing results
        st.subheader(f"Financial Analysis: {selected_ticker} - {selected_quarter}")

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
            st.write("**Financial Metrics Summary (Last 6 Quarters):**")
            display_table = analysis_table.copy()

            # Format the values based on metric type
            # New table structure: Metric | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | QoQ Growth % | YoY Growth %
            for idx, row in display_table.iterrows():
                metric_name = row['Metric']

                # Format all quarter columns (skip 'Metric' and growth columns)
                for col in display_table.columns:
                    if col not in ['Metric', 'QoQ Growth %', 'YoY Growth %']:
                        # This is a quarter column, format as number
                        if pd.notna(row[col]) and row[col] != 'N/A':
                            display_table.at[idx, col] = format_number(row[col], metric_name)

                # Format growth columns
                if 'QoQ Growth %' in display_table.columns and pd.notna(row.get('QoQ Growth %')) and row.get('QoQ Growth %') != 'N/A':
                    display_table.at[idx, 'QoQ Growth %'] = format_growth(row['QoQ Growth %'])

                if 'YoY Growth %' in display_table.columns and pd.notna(row.get('YoY Growth %')) and row.get('YoY Growth %') != 'N/A':
                    display_table.at[idx, 'YoY Growth %'] = format_growth(row['YoY Growth %'])

            st.dataframe(display_table, use_container_width=True, hide_index=True)

            # Display market share table
            if not market_share_table.empty:
                st.write("**Market Share:**")
                st.dataframe(market_share_table, use_container_width=True, hide_index=True)

            # Display investment composition table
            if not investment_composition_table.empty:
                st.write("**Investment Book Composition:**")
                st.dataframe(investment_composition_table, use_container_width=True, hide_index=True)

            # Display prop holdings table
            if not prop_holdings_table.empty:
                st.write("**Top Proprietary Holdings:**")
                st.dataframe(prop_holdings_table, use_container_width=True, hide_index=True)

            # Show raw metrics for debugging (expandable)
            with st.expander("Detailed Calculation Data"):
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

            # Check if cached analysis exists (just show status, don't display)
            if cache_exists:
                try:
                    cached_analysis = cache_df[
                        (cache_df['TICKER'] == selected_ticker) &
                        (cache_df['QUARTER'] == selected_quarter)
                    ]
                    if not cached_analysis.empty:
                        latest_cached = cached_analysis.iloc[-1]
                        generated_date = pd.to_datetime(latest_cached['GENERATED_DATE']).strftime('%Y-%m-%d %H:%M')
                        st.info(f"💾 Cached analysis available (Generated: {generated_date}). Click 'Generate Analysis' to view or 'View All Cached' to browse all.")
                    else:
                        st.info(f"💡 No cached analysis found for {selected_ticker} - {selected_quarter}. Click 'Generate Analysis' to create one.")
                except Exception as e:
                    st.warning(f"Could not check cache: {e}")

        else:
            st.warning(f"Could not calculate metrics for {selected_ticker} in {selected_quarter}")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# Initialize session state for viewing cache
if 'show_cache' not in st.session_state:
    st.session_state.show_cache = False

# Generation controls
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    generate_button = st.button("Generate Analysis", type="primary")
    if generate_button:
        st.session_state.show_cache = False  # Hide cache when generating

with col2:
    if cache_exists:
        if st.button("View All Cached"):
            st.session_state.show_cache = True  # Toggle cache view

with col3:
    # Clear cache button - only for current ticker/quarter
    if cache_exists and selected_ticker and selected_quarter:
        # Check if there's a cached entry for this specific ticker/quarter
        try:
            cache_df = pd.read_csv(cache_file)
            has_cache_entry = not cache_df[
                (cache_df['TICKER'] == selected_ticker) &
                (cache_df['QUARTER'] == selected_quarter)
            ].empty
        except:
            has_cache_entry = False

        if has_cache_entry:
            if st.button("🗑️ Clear This Cache", help=f"Delete cached analysis for {selected_ticker} - {selected_quarter}"):
                try:
                    cache_df = pd.read_csv(cache_file)
                    # Remove entries for this specific ticker and quarter
                    cache_df = cache_df[~(
                        (cache_df['TICKER'] == selected_ticker) &
                        (cache_df['QUARTER'] == selected_quarter)
                    )]
                    # Save back to file
                    cache_df.to_csv(cache_file, index=False)
                    st.success(f"✅ Cleared cache for {selected_ticker} - {selected_quarter}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")

# Generate commentary
if generate_button and selected_ticker and selected_quarter:
    try:
        # Load data for selected ticker/quarter
        ticker_data = load_ticker_data(selected_ticker, selected_quarter)

        if ticker_data.empty:
            st.error(f"No financial data available for {selected_ticker} in {selected_quarter}")
        else:
            # Prepare analysis data
            ticker_data, pivot_data = filter_ticker_data(ticker_data, selected_ticker)
            calculated_metrics = calculate_financial_metrics(ticker_data, selected_quarter, selected_ticker)
            analysis_table = create_analysis_table(ticker_data, calculated_metrics, selected_quarter)
            market_share_table, prop_holdings_table, investment_composition_table = create_summary_tables(selected_ticker, selected_quarter, ticker_data)

            if analysis_table.empty:
                st.error(f"No financial data available for {selected_ticker} in {selected_quarter}")
            else:
                with st.spinner(f"🤖 Generating AI commentary for {selected_ticker} - {selected_quarter}..."):
                    try:
                        # Convert analysis table to string for OpenAI
                        analysis_text = analysis_table.to_string(index=False)

                        # Generate commentary and get the prompt
                        result = generate_commentary(
                            ticker=selected_ticker,
                            year_quarter=selected_quarter,
                            df=ticker_data,  # Pass the ticker-specific data
                            model=model_choice,
                            force_regenerate=force_regenerate,
                            analysis_table=analysis_table,  # Pass the pre-built analysis table
                            market_share_table=market_share_table,  # Pass market share table
                            prop_holdings_table=prop_holdings_table,  # Pass prop holdings table
                            return_prompt=True  # Request the prompt to be returned
                        )

                        # Unpack result
                        commentary, full_prompt = result

                        if commentary.startswith("Error"):
                            st.error(commentary)
                            st.info("💡 **Tips:**\n- Check your OpenAI API key in .streamlit/secrets.toml file\n- Ensure you have API credits\n- Try a different model")
                        else:
                            st.success("Analysis generated successfully!")

                            # Display the generated commentary
                            st.subheader(f"AI Analysis: {selected_ticker} - {selected_quarter}")

                            # Format the commentary to ensure bullets are on separate lines and headers are normal size
                            formatted_commentary = commentary.replace('• ', '\n• ').strip()
                            # Convert markdown headers (## or #) to bold text to keep same text size
                            import re
                            formatted_commentary = re.sub(r'^##\s+', '**', formatted_commentary, flags=re.MULTILINE)
                            formatted_commentary = re.sub(r'^#\s+', '**', formatted_commentary, flags=re.MULTILINE)
                            # Add closing bold tag at end of header lines
                            formatted_commentary = re.sub(r'\*\*(\d+)\.\s+([^\n]+)', r'**\1. \2**', formatted_commentary)

                            st.markdown(formatted_commentary)

                            st.caption(f"Generated with {model_choice} on {datetime.now().strftime('%Y-%m-%d %H:%M')}")

                            # Display the full prompt sent to OpenAI
                            with st.expander("📝 View Full Prompt Sent to OpenAI"):
                                st.markdown("### System Message:")
                                st.code("You are an expert financial analyst specializing in Vietnamese securities and brokerage firms. You MUST follow the exact structure provided in the prompt. Do not deviate from the requested format.")

                                st.markdown("### User Prompt:")
                                st.text(full_prompt)

                                st.caption(f"Model: {model_choice} | Max Tokens: 800 | Temperature: 0.5")

                    except Exception as e:
                        st.error(f"Error generating commentary: {e}")
                        st.info("💡 **Common issues:**\n- Missing OpenAI API key\n- Invalid API key\n- Insufficient API credits\n- Network connectivity")

    except Exception as e:
        st.error(f"Error preparing data for analysis: {e}")

# View cached commentaries (using session state to persist view)
if st.session_state.show_cache and cache_exists:
    st.subheader("📚 Cached AI Commentaries")

    # Add close button
    if st.button("✖ Close Cache View"):
        st.session_state.show_cache = False
        st.rerun()

    try:
        cache_df = pd.read_csv(cache_file)
        cache_df['GENERATED_DATE'] = pd.to_datetime(cache_df['GENERATED_DATE']).dt.strftime('%Y-%m-%d %H:%M')

        # Sort by date descending for most recent first
        cache_df = cache_df.sort_values('GENERATED_DATE', ascending=False).reset_index(drop=True)

        # Display summary
        st.write(f"**Total cached analyses:** {len(cache_df)}")

        # Show recent analyses table
        display_df = cache_df[['TICKER', 'QUARTER', 'GENERATED_DATE', 'MODEL']]
        st.dataframe(display_df, use_container_width=True)

        # Allow selection and viewing of specific cached commentary
        if len(cache_df) > 0:
            st.subheader("View Specific Commentary")

            # Create options list for selectbox
            options_list = [
                f"{row['TICKER']} - {row['QUARTER']} ({row['GENERATED_DATE']})"
                for _, row in cache_df.iterrows()
            ]

            selected_option = st.selectbox(
                "Select cached analysis:",
                options=options_list,
                key="cached_commentary_selector"
            )

            # Find the selected index
            selected_index = options_list.index(selected_option)
            selected_row = cache_df.iloc[selected_index]

            st.subheader(f"Analysis: {selected_row['TICKER']} - {selected_row['QUARTER']}")

            # Format the commentary
            commentary = selected_row['COMMENTARY']
            formatted_commentary = commentary.replace('• ', '\n• ').strip()
            import re
            formatted_commentary = re.sub(r'^##\s+', '**', formatted_commentary, flags=re.MULTILINE)
            formatted_commentary = re.sub(r'^#\s+', '**', formatted_commentary, flags=re.MULTILINE)
            formatted_commentary = re.sub(r'\*\*(\d+)\.\s+([^\n]+)', r'**\1. \2**', formatted_commentary)

            st.markdown(formatted_commentary)
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
        - NPAT (QoQ, YoY growth)
        - Margin Balance (QoQ, YoY growth)
        - ROE (no growth %, shows historical values) - annualized for quarterly data

        **Available for Addition:**
        - ROA (Return on Assets)
        - NET_TRADING_INCOME
        - FEE_INCOME
        - INTEREST_INCOME
        - SGA
        - INTEREST_EXPENSE
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