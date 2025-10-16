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
        'Net Brokerage Income': 'Net_Brokerage_Income',
        'IB Income': 'Net_IB_Income',
        'Margin Income': 'Net_Margin_lending_Income',  # Correct METRIC_CODE for margin lending income
        'Investment Income': 'Net_investment_income',
        'Other Incomes': 'Net_Other_Income',
        'Total Operating Income': 'Total_Operating_Income',  # Total operating income
        'PBT': 'PBT',  # KEYCODE in database
        'NPAT': 'NPAT',  # KEYCODE in database
        'SG&A': 'SG_A',  # Selling, General & Administrative expenses
        'Interest Expense': 'Interest_Expense',  # Interest expense
        'Borrowing Balance': 'Borrowing_Balance',  # Total borrowing
        'Margin Balance': 'Margin_Lending_book',
        'ROE': 'ROE',  # Use existing ROE calculation from CSV
        'CIR': 'CIR',  # Cost-to-Income Ratio (calculated)
        'Interest Rate': 'Interest_Rate'  # Interest rate (calculated)
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
    display_metrics = [
        'Net Brokerage Income',
        'Market Liquidity (Avg Daily)',
        'Trading Value',
        'Brokerage Market Share',
        'Net Brokerage Fee',
        'IB Income',
        'Margin Income',
        'Margin Balance',
        'Margin/Equity %',
        'Margin Lending Rate',
        'Margin Lending Spread',
        'Investment Income',
        'MTM Equities',
        'Non-MTM Equities',
        'Bonds',
        'CDs/Deposits',
        'Other Incomes',
        'Total Operating Income',
        'SG&A',
        'CIR',
        'Interest Expense',
        'Borrowing Balance',
        'Interest Rate',
        'PBT',
        'NPAT',
        'ROE'
    ]

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
                        margin_balance_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Margin_Lending_book')
                    if total_equity_value is None:
                        total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'BS.142')

                    if total_equity_value and total_equity_value != 0:
                        ratio = (margin_balance_value / total_equity_value) * 100
                        quarter_values.append(ratio)
                    else:
                        quarter_values.append(0)
                continue

            if metric_name == 'CIR':
                # Calculate CIR = SG&A / (Total Operating Income - Investment Income)
                sga = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'SG_A')
                total_op_income = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Total_Operating_Income')
                investment_income = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Net_investment_income')

                denominator = total_op_income - investment_income
                if denominator and denominator != 0:
                    cir = abs(sga) / denominator * 100  # Use abs since SGA is negative
                    quarter_values.append(cir)
                else:
                    quarter_values.append(0)
                continue

            if metric_name == 'Interest Rate':
                # Calculate Interest Rate = Interest Expense / Average Borrowing Balance * 100
                # For quarterly data, annualize by multiplying by 4
                interest_expense = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Interest_Expense')
                borrowing_balance = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Borrowing_Balance')

                # Get previous quarter borrowing for average
                quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q) and q != ''])
                current_quarter_label = f"{quarter_num}Q{str(year)[-2:]}"
                if current_quarter_label in quarters:
                    current_idx = quarters.index(current_quarter_label)
                    if current_idx > 0:
                        prev_quarter_label = quarters[current_idx - 1]
                        # Parse previous quarter
                        try:
                            prev_quarter_num = int(prev_quarter_label[0])
                            prev_year_str = prev_quarter_label[-2:]
                            prev_year = 2000 + int(prev_year_str) if int(prev_year_str) < 50 else 1900 + int(prev_year_str)
                            prev_borrowing = get_calc_metric_value(ticker_data, ticker, prev_year, prev_quarter_num, 'Borrowing_Balance')
                            avg_borrowing = (borrowing_balance + prev_borrowing) / 2 if prev_borrowing else borrowing_balance
                        except:
                            avg_borrowing = borrowing_balance
                    else:
                        avg_borrowing = borrowing_balance
                else:
                    avg_borrowing = borrowing_balance

                if avg_borrowing and avg_borrowing != 0:
                    # Annualize the rate for quarterly data
                    interest_rate = abs(interest_expense) / avg_borrowing * 100 * 4
                    quarter_values.append(interest_rate)
                else:
                    quarter_values.append(0)
                continue

            if metric_name == 'Trading Value':
                # Calculate Trading Value = Institution shares + Investor shares trading value (in billions VND)
                institution_shares = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Institution_shares_trading_value')
                investor_shares = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Investor_shares_trading_value')
                total_trading_value = (institution_shares + investor_shares) / 1_000_000_000  # Convert to billions
                quarter_values.append(total_trading_value)
                continue

            if metric_name == 'Brokerage Market Share':
                # First, try to get market share from HSX API (for Top 10 brokers)
                # Reconstruct quarter_label from year and quarter_num (e.g., "1Q24")
                quarter_label = f"{quarter_num}Q{str(year)[-2:]}"
                hsx_data = fetch_market_share(ticker, quarter_label)

                # If HSX API returns data (broker is in Top 10), use it
                if hsx_data['market_share'] > 0:
                    # Use HSX-provided market share (already in percentage)
                    quarter_values.append(hsx_data['market_share'])
                else:
                    # Calculate Market Share for brokers not in Top 10
                    # Formula: Trading Value / (Market Liquidity * Trading Days in Quarter) / 2
                    institution_shares = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Institution_shares_trading_value')
                    investor_shares = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Investor_shares_trading_value')
                    total_trading_value = institution_shares + investor_shares

                    # Get market liquidity and trading days
                    if not market_liquidity_df.empty:
                        liquidity_row = market_liquidity_df[
                            (market_liquidity_df['Year'] == year) &
                            (market_liquidity_df['Quarter'] == quarter_num)
                        ]
                        if not liquidity_row.empty:
                            avg_daily_turnover_bn = liquidity_row.iloc[0]['Avg Daily Turnover (B VND)']
                            trading_days = liquidity_row.iloc[0]['Trading Days']

                            # Market liquidity is in billions, convert to VND for calculation
                            total_market_value = avg_daily_turnover_bn * 1_000_000_000 * trading_days

                            if total_market_value and total_market_value != 0:
                                market_share = (total_trading_value / total_market_value) / 2 * 100  # Divide by 2 and convert to percentage
                                quarter_values.append(market_share)
                            else:
                                quarter_values.append(0)
                        else:
                            quarter_values.append(0)
                    else:
                        quarter_values.append(0)
                continue

            if metric_name == 'Net Brokerage Fee':
                # Calculate Net Brokerage Fee = Net Brokerage Income / Trading Value (in basis points)
                net_brokerage_income = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Net_Brokerage_Income')
                institution_shares = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Institution_shares_trading_value')
                investor_shares = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Investor_shares_trading_value')
                total_trading_value = institution_shares + investor_shares

                if total_trading_value and total_trading_value != 0:
                    # Calculate as basis points (bps): (income / trading value) * 10000
                    net_brokerage_fee_bps = (net_brokerage_income / total_trading_value) * 10000
                    quarter_values.append(net_brokerage_fee_bps)
                else:
                    quarter_values.append(0)
                continue

            metric_code = {
                'Net Brokerage Income': 'Net_Brokerage_Income',
                'IB Income': 'Net_IB_Income',
                'Margin Income': 'Net_Margin_lending_Income',  # Correct METRIC_CODE for margin lending income
                'Investment Income': 'Net_investment_income',
                'MTM Equities': 'mtm_equities_market_value',
                'Non-MTM Equities': 'not_mtm_equities_market_value',
                'Bonds': 'bonds_market_value',
                'CDs/Deposits': 'cds_deposits_market_value',
                'Other Incomes': 'Net_Other_Income',
                'Total Operating Income': 'Total_Operating_Income',
                'PBT': 'PBT',  # KEYCODE in database
                'NPAT': 'NPAT',  # KEYCODE in database
                'SG&A': 'SG_A',
                'Interest Expense': 'Interest_Expense',
                'Borrowing Balance': 'Borrowing_Balance',
                'Margin Balance': 'Margin_Lending_book',
                'Margin Lending Rate': 'MARGIN_LENDING_RATE',
                'Margin Lending Spread': 'MARGIN_LENDING_SPREAD',
                'ROE': 'ROE',
                'CIR': 'CIR',
                'Interest Rate': 'Interest_Rate'
            }.get(metric_name)

            value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, metric_code)
            quarter_values.append(value)

            # Store for ratio calculation
            if metric_name == 'Margin Balance':
                margin_balance_value = value
                # Also get Total Equity for the Margin/Equity % calculation
                total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'BS.142')
            elif metric_name == 'ROE':
                # Also get Total Equity for the ratio if not already fetched
                if total_equity_value is None:
                    total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'BS.142')

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
    # Special handling for calculated metrics that don't exist in database
    if metric_code == 'ROE':
        # Calculate ROE = NPAT / Total Equity * 100 (annualized for quarterly)
        npat = get_calc_metric_value(df, ticker, year, quarter, 'NPAT')
        equity = get_calc_metric_value(df, ticker, year, quarter, 'BS.142')  # Use actual KEYCODE
        if equity and equity != 0:
            roe = (npat / equity) * 100
            # Annualize for quarterly data (multiply by 4)
            if quarter in [1, 2, 3, 4]:
                roe = roe * 4
            return roe
        return 0

    # Base filter
    base_filter = (
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter)
    )

    result = pd.DataFrame()

    # Try KEYCODE directly (for codes like 'PBT', 'NPAT', 'BS.142')
    result = df[base_filter & (df['KEYCODE'] == metric_code)]

    # If empty, try METRIC_CODE with STATEMENT_TYPE='CALC'
    if result.empty:
        result = df[base_filter & (df['STATEMENT_TYPE'] == 'CALC') & (df['METRIC_CODE'] == metric_code)]

    # If still empty, try METRIC_CODE without STATEMENT_TYPE filter (for balance sheet items)
    if result.empty:
        result = df[base_filter & (df['METRIC_CODE'] == metric_code)]

    if len(result) > 0:
        value = result.iloc[0]['VALUE']
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
    """Get investment book composition with simplified 4-category structure for selected quarter"""
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

    if not investment_data or not any(value > 0 for value in investment_data.values()):
        return pd.DataFrame()

    # Build simplified composition with 4 categories
    composition = []
    total_value = sum(investment_data.values())

    for category in ['MTM Equities', 'Non-MTM Equities', 'Bonds', 'CDs/Deposits']:
        value = investment_data.get(category, 0)
        
        if value > 0:
            composition.append({
                'Investment Type': category,
                'Value (B VND)': f"{value / 1_000_000_000:,.1f}",
                'Composition %': f"{(value / total_value * 100):.1f}%"
            })

    # Add total row
    if composition:
        composition.append({
            'Investment Type': 'Total Investments',
            'Value (B VND)': f"{total_value / 1_000_000_000:,.1f}",
            'Composition %': '100.0%'
        })

    return pd.DataFrame(composition)

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

        # Display TOI drivers analysis
        st.write("**TOI Drivers Analysis:**")
        try:
            from utils.toi_drivers import calculate_toi_drivers

            # Create tabs for QoQ and YoY
            tab_qoq, tab_yoy = st.tabs(["Quarter-over-Quarter", "Year-over-Year"])

            with tab_qoq:
                drivers_qoq = calculate_toi_drivers(selected_ticker, selected_quarter, 'QoQ')

                if not drivers_qoq.empty:
                    # Format the dataframe for display
                    display_df = drivers_qoq.copy()
                    for idx, row in display_df.iterrows():
                        # Skip section headers (they have empty string values)
                        if row['Current'] == '':
                            continue
                        # Format numeric values
                        display_df.at[idx, 'Current'] = f"{row['Current']:.1f}B"
                        display_df.at[idx, 'Prior'] = f"{row['Prior']:.1f}B"
                        display_df.at[idx, 'Change'] = f"{row['Change']:+.1f}B"
                        display_df.at[idx, 'Impact (pp)'] = f"{row['Impact (pp)']:+.1f}pp"
                        if '% of TOI' in display_df.columns and row['% of TOI'] != '':
                            display_df.at[idx, '% of TOI'] = f"{row['% of TOI']:.1f}%"

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Show summary metrics
                    growth_pct = drivers_qoq.attrs.get('growth_pct', 0)
                    prior_q = drivers_qoq.attrs.get('prior_quarter', '')
                    st.info(f"TOI Growth: **{growth_pct:+.1f}%** vs {prior_q}")
                else:
                    st.warning("Insufficient data for QoQ analysis (need at least 2 quarters)")

            with tab_yoy:
                drivers_yoy = calculate_toi_drivers(selected_ticker, selected_quarter, 'YoY')

                if not drivers_yoy.empty:
                    # Format the dataframe for display
                    display_df = drivers_yoy.copy()
                    for idx, row in display_df.iterrows():
                        # Skip section headers (they have empty string values)
                        if row['Current'] == '':
                            continue
                        # Format numeric values
                        display_df.at[idx, 'Current'] = f"{row['Current']:.1f}B"
                        display_df.at[idx, 'Prior'] = f"{row['Prior']:.1f}B"
                        display_df.at[idx, 'Change'] = f"{row['Change']:+.1f}B"
                        display_df.at[idx, 'Impact (pp)'] = f"{row['Impact (pp)']:+.1f}pp"
                        if '% of TOI' in display_df.columns and row['% of TOI'] != '':
                            display_df.at[idx, '% of TOI'] = f"{row['% of TOI']:.1f}%"

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                    # Show summary metrics
                    growth_pct = drivers_yoy.attrs.get('growth_pct', 0)
                    prior_q = drivers_yoy.attrs.get('prior_quarter', '')
                    st.info(f"TOI Growth: **{growth_pct:+.1f}%** vs {prior_q}")
                else:
                    st.warning("Insufficient data for YoY analysis (need at least 5 quarters)")

        except Exception as e:
            import traceback
            st.error(f"Error calculating TOI drivers: {e}")
            st.code(traceback.format_exc())

        # Show calculated metrics
        if not calculated_metrics.empty and not analysis_table.empty:
            # Format numbers for display
            def format_number(value, metric_name):
                if pd.isna(value) or value == 'N/A':
                    return "N/A"
                try:
                    value = float(value)
                    # Percentages - already calculated as percentages (e.g., 15.5 means 15.5%)
                    if metric_name in ['ROE', 'ROA', 'Margin/Equity %', 'CIR', 'Interest Rate', 'Brokerage Market Share', 'Margin Lending Rate', 'Margin Lending Spread']:
                        return f"{value:.2f}%"
                    # Basis points
                    elif metric_name == 'Net Brokerage Fee':
                        return f"{value:.2f} bps"
                    # Trading Value already in billions
                    elif metric_name == 'Trading Value':
                        return f"{value:,.1f}B VND"
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

                        # Calculate TOI drivers for OpenAI
                        from utils.toi_drivers import calculate_toi_drivers
                        drivers_qoq = calculate_toi_drivers(selected_ticker, selected_quarter, 'QoQ')
                        drivers_yoy = calculate_toi_drivers(selected_ticker, selected_quarter, 'YoY')

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
                            investment_composition_table=investment_composition_table,  # Pass investment composition table
                            toi_drivers_qoq=drivers_qoq,  # Pass TOI drivers QoQ
                            toi_drivers_yoy=drivers_yoy,  # Pass TOI drivers YoY
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