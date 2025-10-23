"""
Historical Financial Analysis with Comprehensive Metrics
Shows detailed financial metrics, market share, prop holdings, and investment composition
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Historical Financial Analysis",
    layout="wide"
)

import pandas as pd
import toml
import requests

# Load theme from config.toml
theme_config = toml.load("utils/config.toml")
theme = theme_config["theme"]
primary_color = theme["primaryColor"]

# Import utilities
from utils.brokerage_codes import BROKERAGE_CODE_MAP

def get_display_ticker(data_ticker):
    """Convert data ticker (TCBS) to display ticker (TCX) for UI consistency"""
    # Find the display code that maps to this data ticker
    for display_code, data_code in BROKERAGE_CODE_MAP.items():
        if data_code == data_ticker:
            return display_code
    return data_ticker  # Return original if no mapping found

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
        'Total Debt': 'Total_Debt_Balance',  # Total debt/borrowing balance
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
    """Step 3: Combine historical data and calculated metrics for analysis - Last 6 Quarters

    Returns:
        tuple: (df_income_statement, df_balance_sheet) - Two separate DataFrames
    """

    # Get all quarters sorted chronologically
    quarters = sort_quarters_chronologically([q for q in ticker_data['QUARTER_LABEL'].unique() if pd.notna(q) and q != ''])

    # Find the index of selected quarter and get last 6 quarters
    if selected_quarter not in quarters:
        return pd.DataFrame(), pd.DataFrame()

    current_idx = quarters.index(selected_quarter)
    # Get last 6 quarters including current (or fewer if not available)
    last_6_quarters = quarters[max(0, current_idx - 5):current_idx + 1]

    if len(last_6_quarters) == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Get ticker from ticker_data
    if ticker_data.empty or 'TICKER' not in ticker_data.columns:
        return pd.DataFrame(), pd.DataFrame()

    ticker = ticker_data['TICKER'].iloc[0]

    # Load market liquidity data
    market_liquidity_df = load_market_liquidity_data()

    # Income Statement metrics
    income_statement_metrics = [
        'Net Brokerage Income',
        'Market Liquidity (Avg Daily)',
        'Brokerage Market Share',
        'Net Brokerage Fee',
        'IB Income',
        'Margin Income',
        'Margin Lending Rate',
        'Margin Lending Spread',
        'Investment Income',
        'Other Incomes',
        'Total Operating Income',
        'SG&A',
        'CIR',
        'PBT',
        'NPAT',
        'ROE'
    ]

    # Balance Sheet metrics - organized as Assets, then Liabilities & Equity
    balance_sheet_metrics = [
        'Margin Balance',
        'MTM Equities',
        'Non-MTM Equities',
        'Bonds',
        'CDs/Deposits',
        'Total Investments',
        'Total Assets',
        'Total Debt',
        'Interest Expense',
        'Interest Rate',
        'Total Equity',
        'Margin/Equity %'
    ]

    # Create two separate table structures
    income_statement_data = {'Metric': income_statement_metrics}
    balance_sheet_data = {'Metric': balance_sheet_metrics}

    # For each of the last 6 quarters, get the metric values
    for quarter in last_6_quarters:
        # Parse quarter to get year and quarter_num
        try:
            quarter_num = int(quarter[0])
            year_str = quarter[-2:]
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        except:
            continue

        # Get metrics for this quarter - Income Statement
        income_quarter_values = []
        # Get metrics for Balance Sheet
        balance_quarter_values = []

        # Variables for calculations
        margin_balance_value = None
        total_equity_value = None

        # Asset components for Total Assets calculation
        mtm_equities_value = 0
        non_mtm_equities_value = 0
        bonds_value = 0
        cds_deposits_value = 0

        # Process Income Statement metrics
        for metric_name in income_statement_metrics:
            if metric_name == 'Market Liquidity (Avg Daily)':
                # Get market liquidity for this quarter
                if not market_liquidity_df.empty:
                    liquidity_row = market_liquidity_df[
                        (market_liquidity_df['Year'] == year) &
                        (market_liquidity_df['Quarter'] == quarter_num)
                    ]
                    if not liquidity_row.empty:
                        income_quarter_values.append(liquidity_row.iloc[0]['Avg Daily Turnover (B VND)'])
                    else:
                        income_quarter_values.append(0)
                else:
                    income_quarter_values.append(0)
                continue

            if metric_name == 'CIR':
                # Calculate CIR = SG&A / (Total Operating Income - Investment Income)
                sga = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'SG_A')
                total_op_income = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Total_Operating_Income')
                investment_income = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Net_investment_income')

                denominator = total_op_income - investment_income
                if denominator and denominator != 0:
                    cir = abs(sga) / denominator * 100  # Use abs since SGA is negative
                    income_quarter_values.append(cir)
                else:
                    income_quarter_values.append(0)
                continue

            if metric_name == 'Brokerage Market Share':
                # First, try to get market share from HSX API (for Top 10 brokers)
                # Reconstruct quarter_label from year and quarter_num (e.g., "1Q24")
                quarter_label = f"{quarter_num}Q{str(year)[-2:]}"
                hsx_data = fetch_market_share(ticker, quarter_label)

                # If HSX API returns data (broker is in Top 10), use it
                if hsx_data['market_share'] > 0:
                    # Use HSX-provided market share (already in percentage)
                    income_quarter_values.append(hsx_data['market_share'])
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
                                income_quarter_values.append(market_share)
                            else:
                                income_quarter_values.append(0)
                        else:
                            income_quarter_values.append(0)
                    else:
                        income_quarter_values.append(0)
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
                    income_quarter_values.append(net_brokerage_fee_bps)
                else:
                    income_quarter_values.append(0)
                continue

            # Standard metric code mapping for Income Statement
            metric_code = {
                'Net Brokerage Income': 'Net_Brokerage_Income',
                'IB Income': 'Net_IB_Income',
                'Margin Income': 'Net_Margin_lending_Income',
                'Investment Income': 'Net_investment_income',
                'Other Incomes': 'Net_Other_Income',
                'Total Operating Income': 'Total_Operating_Income',
                'PBT': 'PBT',
                'NPAT': 'NPAT',
                'SG&A': 'SG_A',
                'Margin Lending Rate': 'MARGIN_LENDING_RATE',
                'Margin Lending Spread': 'MARGIN_LENDING_SPREAD',
                'ROE': 'ROE'
            }.get(metric_name)

            if metric_code:
                value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, metric_code)
                income_quarter_values.append(value)

        # Process Balance Sheet metrics
        for metric_name in balance_sheet_metrics:
            if metric_name == 'Total Investments':
                # Calculate Total Investments = sum of investment book items
                total_investments = mtm_equities_value + non_mtm_equities_value + bonds_value + cds_deposits_value
                balance_quarter_values.append(total_investments)
                continue

            if metric_name == 'Total Assets':
                # Calculate Total Assets = Margin Balance + MTM Equities + Non-MTM Equities + Bonds + CDs/Deposits
                # Fetch margin balance if not already fetched
                if margin_balance_value is None:
                    margin_balance_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Margin_Lending_book')

                total_assets = margin_balance_value + mtm_equities_value + non_mtm_equities_value + bonds_value + cds_deposits_value
                balance_quarter_values.append(total_assets)
                continue

            if metric_name == 'Total Equity':
                # Get Total Equity using 'BS.142'
                total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'BS.142')
                balance_quarter_values.append(total_equity_value)
                continue

            if metric_name == 'Margin/Equity %':
                # Calculate margin/equity ratio
                if margin_balance_value is None:
                    margin_balance_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'Margin_Lending_book')
                if total_equity_value is None:
                    total_equity_value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'BS.142')

                if total_equity_value and total_equity_value != 0:
                    ratio = (margin_balance_value / total_equity_value) * 100
                    balance_quarter_values.append(ratio)
                else:
                    balance_quarter_values.append(0)
                continue

            if metric_name == 'Interest Rate':
                # Get Interest Rate directly from database using INTEREST_RATE keycode
                interest_rate = get_calc_metric_value(ticker_data, ticker, year, quarter_num, 'INTEREST_RATE')
                balance_quarter_values.append(interest_rate)
                continue

            # Standard metric code mapping for Balance Sheet
            metric_code = {
                'Margin Balance': 'Margin_Lending_book',
                'MTM Equities': 'mtm_equities_market_value',
                'Non-MTM Equities': 'not_mtm_equities_market_value',
                'Bonds': 'bonds_market_value',
                'CDs/Deposits': 'cds_deposits_market_value',
                'Total Debt': 'Total_Debt_Balance',
                'Interest Expense': 'Interest_Expense'
            }.get(metric_name)

            if metric_code:
                value = get_calc_metric_value(ticker_data, ticker, year, quarter_num, metric_code)
                balance_quarter_values.append(value)

                # Store asset components for Total Assets calculation
                if metric_name == 'Margin Balance':
                    margin_balance_value = value
                elif metric_name == 'MTM Equities':
                    mtm_equities_value = value
                elif metric_name == 'Non-MTM Equities':
                    non_mtm_equities_value = value
                elif metric_name == 'Bonds':
                    bonds_value = value
                elif metric_name == 'CDs/Deposits':
                    cds_deposits_value = value

        # Add this quarter's columns to both tables
        income_statement_data[quarter] = income_quarter_values
        balance_sheet_data[quarter] = balance_quarter_values

    # Create DataFrames
    df_income_statement = pd.DataFrame(income_statement_data)
    df_balance_sheet = pd.DataFrame(balance_sheet_data)

    # Check if we have any quarter columns besides 'Metric'
    if len(df_income_statement.columns) <= 1 or len(df_balance_sheet.columns) <= 1:
        return pd.DataFrame(), pd.DataFrame()

    # Add growth columns for Income Statement
    if len(last_6_quarters) >= 2 and selected_quarter in df_income_statement.columns:
        prev_quarter = last_6_quarters[-2]
        if prev_quarter in df_income_statement.columns:
            qoq_growth = []
            for metric in income_statement_metrics:
                if metric in ['ROE']:
                    qoq_growth.append('N/A')
                else:
                    try:
                        current_val = df_income_statement[df_income_statement['Metric'] == metric][selected_quarter].values[0]
                        prev_val = df_income_statement[df_income_statement['Metric'] == metric][prev_quarter].values[0]
                        if prev_val != 0 and prev_val != 'N/A' and current_val != 'N/A':
                            growth = ((current_val - prev_val) / abs(prev_val)) * 100
                            qoq_growth.append(growth)
                        else:
                            qoq_growth.append('N/A')
                    except (IndexError, KeyError):
                        qoq_growth.append('N/A')

            df_income_statement['QoQ Growth %'] = qoq_growth

    # Add YoY growth for Income Statement if we have at least 5 quarters
    if len(last_6_quarters) >= 5:
        yoy_quarter = last_6_quarters[-5]  # 4 quarters ago
        if yoy_quarter in df_income_statement.columns and selected_quarter in df_income_statement.columns:
            yoy_growth = []
            for metric in income_statement_metrics:
                if metric in ['ROE']:
                    yoy_growth.append('N/A')
                else:
                    try:
                        current_val = df_income_statement[df_income_statement['Metric'] == metric][selected_quarter].values[0]
                        yoy_val = df_income_statement[df_income_statement['Metric'] == metric][yoy_quarter].values[0]
                        if yoy_val != 0 and yoy_val != 'N/A' and current_val != 'N/A':
                            growth = ((current_val - yoy_val) / abs(yoy_val)) * 100
                            yoy_growth.append(growth)
                        else:
                            yoy_growth.append('N/A')
                    except (IndexError, KeyError):
                        yoy_growth.append('N/A')

            df_income_statement['YoY Growth %'] = yoy_growth

    # Add growth columns for Balance Sheet
    if len(last_6_quarters) >= 2 and selected_quarter in df_balance_sheet.columns:
        prev_quarter = last_6_quarters[-2]
        if prev_quarter in df_balance_sheet.columns:
            qoq_growth = []
            for metric in balance_sheet_metrics:
                # Skip ratio/percentage metrics for growth calculation
                if metric in ['Margin/Equity %', 'Interest Rate']:
                    qoq_growth.append('N/A')
                else:
                    try:
                        current_val = df_balance_sheet[df_balance_sheet['Metric'] == metric][selected_quarter].values[0]
                        prev_val = df_balance_sheet[df_balance_sheet['Metric'] == metric][prev_quarter].values[0]
                        if prev_val != 0 and prev_val != 'N/A' and current_val != 'N/A':
                            growth = ((current_val - prev_val) / abs(prev_val)) * 100
                            qoq_growth.append(growth)
                        else:
                            qoq_growth.append('N/A')
                    except (IndexError, KeyError):
                        qoq_growth.append('N/A')

            df_balance_sheet['QoQ Growth %'] = qoq_growth

    # Add YoY growth for Balance Sheet if we have at least 5 quarters
    if len(last_6_quarters) >= 5:
        yoy_quarter = last_6_quarters[-5]  # 4 quarters ago
        if yoy_quarter in df_balance_sheet.columns and selected_quarter in df_balance_sheet.columns:
            yoy_growth = []
            for metric in balance_sheet_metrics:
                # Skip ratio/percentage metrics for growth calculation
                if metric in ['Margin/Equity %', 'Interest Rate']:
                    yoy_growth.append('N/A')
                else:
                    try:
                        current_val = df_balance_sheet[df_balance_sheet['Metric'] == metric][selected_quarter].values[0]
                        yoy_val = df_balance_sheet[df_balance_sheet['Metric'] == metric][yoy_quarter].values[0]
                        if yoy_val != 0 and yoy_val != 'N/A' and current_val != 'N/A':
                            growth = ((current_val - yoy_val) / abs(yoy_val)) * 100
                            yoy_growth.append(growth)
                        else:
                            yoy_growth.append('N/A')
                    except (IndexError, KeyError):
                        yoy_growth.append('N/A')

            df_balance_sheet['YoY Growth %'] = yoy_growth

    return df_income_statement, df_balance_sheet

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
        # Most tickers match directly, but some need mapping
        ticker_to_brokerage_code = {
            'VCI': 'Vietcap',
            'HCM': 'HSC',
            'VND': 'VNDS',
            'FTS': 'FPTS',
            'TCX': 'TCBS'  # Map TCX to TCBS for HSX API
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
def get_all_prop_holdings_last_quarters(ticker, quarters_list):
    """Get ALL proprietary trading holdings from Prop book.xlsx for last 6 quarters"""
    try:
        prop_df = pd.read_excel('sql/Prop book.xlsx')

        # Filter for the specific broker and quarters in the list
        broker_data = prop_df[
            (prop_df['Broker'] == ticker) &
            (prop_df['Quarter'].isin(quarters_list))
        ]

        if broker_data.empty:
            return pd.DataFrame()

        # Exclude PBT and "Other AFS" only - keep "Other FVTPL" and "Others"
        broker_data = broker_data[~broker_data['Ticker'].isin(['PBT', 'Other AFS'])]

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

        # Create a pivot table with Ticker as rows and Quarters as columns
        # Show total value for each ticker in each quarter
        holdings_pivot = broker_data.pivot_table(
            index='Ticker',
            columns='Quarter',
            values='Total_Value',
            aggfunc='sum',
            fill_value=0
        )

        # Sort columns chronologically
        sorted_quarters = sort_quarters_chronologically(holdings_pivot.columns.tolist())
        holdings_pivot = holdings_pivot[sorted_quarters]

        # Sort rows: "Other FVTPL" and "Others" at the bottom, rest sorted by most recent quarter value
        if len(sorted_quarters) > 0:
            most_recent_quarter = sorted_quarters[-1]

            # Separate "Others" entries
            others_rows = holdings_pivot[holdings_pivot.index.isin(['Other FVTPL', 'Others'])]
            regular_rows = holdings_pivot[~holdings_pivot.index.isin(['Other FVTPL', 'Others'])]

            # Sort regular rows by most recent quarter value (descending)
            if not regular_rows.empty:
                regular_rows = regular_rows.sort_values(by=most_recent_quarter, ascending=False)

            # Combine: regular holdings first, then "Others" at bottom
            if not others_rows.empty:
                holdings_pivot = pd.concat([regular_rows, others_rows])
            else:
                holdings_pivot = regular_rows

        return holdings_pivot

    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def calculate_prop_holdings_volume(ticker, quarters_list):
    """
    Calculate volume (number of shares) held for each stock in prop holdings.
    Volume = Market Value / Quarter-End Price

    Args:
        ticker: Broker ticker (e.g., 'SSI', 'VCI')
        quarters_list: List of quarter labels (e.g., ['4Q24', '1Q25', '2Q25'])

    Returns:
        DataFrame with Ticker as rows, Quarters as columns, values in number of shares
        Same structure as prop holdings table for easy comparison
    """
    from utils.stock_prices import get_quarter_end_price

    try:
        # Get the market value holdings data
        holdings_pivot = get_all_prop_holdings_last_quarters(ticker, quarters_list)

        if holdings_pivot.empty:
            return pd.DataFrame()

        # Create volume pivot with same structure
        volume_data = {}

        for quarter in holdings_pivot.columns:
            quarter_volumes = []

            for stock_ticker in holdings_pivot.index:
                market_value = holdings_pivot.loc[stock_ticker, quarter]

                # Skip if no holdings or if it's a special category
                if market_value == 0 or pd.isna(market_value):
                    quarter_volumes.append(0)
                    continue

                # Skip "Others" and "Other FVTPL" - can't calculate volume for aggregates
                if stock_ticker.upper() in ['OTHERS', 'OTHER FVTPL']:
                    quarter_volumes.append(0)
                    continue

                # Get quarter-end price (cached)
                quarter_end_price = get_quarter_end_price(stock_ticker, quarter)

                if quarter_end_price and quarter_end_price > 0:
                    # Calculate volume: market value (in billions) / price (in VND)
                    # Market value is in billions VND, so multiply by 1 billion first
                    volume = (market_value * 1_000_000_000) / quarter_end_price
                    quarter_volumes.append(volume)
                else:
                    # Price not available
                    quarter_volumes.append(0)

            volume_data[quarter] = quarter_volumes

        # Create DataFrame with same structure as holdings
        volume_pivot = pd.DataFrame(volume_data, index=holdings_pivot.index)

        return volume_pivot

    except Exception as e:
        return pd.DataFrame()

def get_investment_composition_last_quarters(ticker_data, ticker, quarters_list):
    """Get investment book composition with simplified 4-category structure for last 6 quarters"""
    from utils.investment_book import get_investment_data

    if not quarters_list:
        return pd.DataFrame()

    # Categories to display
    categories = ['MTM Equities', 'Non-MTM Equities', 'Bonds', 'CDs/Deposits']

    # Build data structure: Investment Type as rows, Quarters as columns
    composition_data = {'Investment Type': categories + ['Total Investments']}

    for quarter_label in quarters_list:
        # Parse quarter to get year and quarter number
        try:
            quarter_num = int(quarter_label[0])
            year_str = quarter_label[-2:]
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        except:
            # Add empty column if parsing fails
            composition_data[quarter_label] = ['-'] * len(composition_data['Investment Type'])
            continue

        # Get investment data for this quarter
        investment_data = get_investment_data(ticker_data, ticker, year, quarter_num)

        if not investment_data or not any(value > 0 for value in investment_data.values()):
            # Add empty column if no data
            composition_data[quarter_label] = ['-'] * len(composition_data['Investment Type'])
            continue

        # Calculate values for each category
        quarter_values = []
        total_value = sum(investment_data.values())

        for category in categories:
            value = investment_data.get(category, 0)
            if value > 0:
                quarter_values.append(f"{value / 1_000_000_000:,.1f}")
            else:
                quarter_values.append('-')

        # Add total
        quarter_values.append(f"{total_value / 1_000_000_000:,.1f}")

        composition_data[quarter_label] = quarter_values

    return pd.DataFrame(composition_data)

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

    # Build market share table for last 6 quarters (both HSX API and calculated)
    market_share_table = pd.DataFrame()
    market_data = {'Quarter': []}
    market_share_values = []
    market_rank_values = []
    market_liquidity_df = load_market_liquidity_data()

    for quarter in last_6_quarters:
        # Parse quarter to get year and quarter_num
        try:
            quarter_num = int(quarter[0])
            year_str = quarter[-2:]
            year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
        except:
            market_data['Quarter'].append(quarter)
            market_share_values.append('N/A')
            market_rank_values.append('N/A')
            continue

        market_data['Quarter'].append(quarter)

        # Try HSX API first
        market_share_data = fetch_market_share(ticker, quarter)

        if market_share_data['market_share'] > 0:
            # Use HSX API data (broker is in Top 10)
            market_share_values.append(f"{market_share_data['market_share']:.2f}%")
            market_rank_values.append(f"#{market_share_data['rank']}" if market_share_data['rank'] else 'N/A')
        else:
            # Calculate market share for brokers outside Top 10
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

                    if total_market_value and total_market_value != 0 and total_trading_value > 0:
                        market_share = (total_trading_value / total_market_value) / 2 * 100
                        market_share_values.append(f"{market_share:.2f}%")
                        market_rank_values.append('N/A')
                    else:
                        market_share_values.append('N/A')
                        market_rank_values.append('N/A')
                else:
                    market_share_values.append('N/A')
                    market_rank_values.append('N/A')
            else:
                market_share_values.append('N/A')
                market_rank_values.append('N/A')

    if market_data['Quarter']:
        # Use the original ticker for display (TCX stays as TCX)
        market_data[f'{ticker} Market Share'] = market_share_values
        market_data[f'{ticker} Rank'] = market_rank_values
        market_share_table = pd.DataFrame(market_data)

    # Build prop holdings table for last 6 quarters - ALL holdings across all quarters
    prop_holdings_table = get_all_prop_holdings_last_quarters(ticker, last_6_quarters)

    # Build investment composition table for last 6 quarters
    investment_composition_table = get_investment_composition_last_quarters(ticker_data, ticker, last_6_quarters)

    return market_share_table, prop_holdings_table, investment_composition_table

# Title and description
st.title("Historical Financial Analysis")

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

# Whitelist of tickers to display in UI (data ticker codes)
# Note: TCBS maps to TCX, VPBS maps to VPX, VPS maps to VCK in display
DISPLAY_TICKERS_WHITELIST = ['SSI', 'VCI', 'VND', 'HCM', 'TCBS', 'VPBS', 'VPS', 'MBS', 'VIX', 'SHS', 'BSI', 'DSE', 'FTS', 'VDS', 'ORS']

# Broker groups for organized display
broker_groups = {
    'Top Tier': ['SSI', 'VCI', 'VND', 'HCM', 'TCBS', 'VPBS', 'VPS'],
    'Mid Tier': ['MBS', 'VIX', 'SHS', 'BSI', 'FTS'],
    'Regional': ['DSE', 'VDS', 'LPBS', 'Kafi', 'ACBS', 'OCBS', 'HDBS'],
}

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

# Main interface
st.subheader("Select Broker and Quarter")

# Input controls
col1, col2 = st.columns(2)

with col1:
    # Create broker list ordered by tier (without showing tier headers)
    ordered_tickers = []

    # Add brokers in tier order
    for group_name, tickers in broker_groups.items():
        for ticker in tickers:
            if ticker in available_tickers and ticker in DISPLAY_TICKERS_WHITELIST:
                ordered_tickers.append(ticker)

    # Add any brokers not in groups at the end (only if whitelisted)
    ungrouped = [t for t in available_tickers if not any(t in group for group in broker_groups.values()) and t in DISPLAY_TICKERS_WHITELIST]
    ordered_tickers.extend(ungrouped)

    # Create display names
    ticker_display_names = [get_display_ticker(ticker) for ticker in ordered_tickers]

    # Find default index (SSI)
    default_ticker = 'SSI'
    default_index = ordered_tickers.index(default_ticker) if default_ticker in ordered_tickers else 0

    selected_ticker_index = st.selectbox(
        "Select Broker:",
        range(len(ordered_tickers)),
        format_func=lambda x: ticker_display_names[x],
        index=default_index,
        help="Choose a broker to analyze"
    )

    # Get the actual ticker code for data queries
    selected_ticker = ordered_tickers[selected_ticker_index]
    # Get the display name for UI
    selected_ticker_display = get_display_ticker(selected_ticker)

with col2:
    # Get quarters available for the selected ticker (lightweight query)
    ticker_quarters = get_ticker_quarters_list(selected_ticker, start_year=2017)

    if ticker_quarters:
        selected_quarter = st.selectbox(
            "Select Quarter:",
            ticker_quarters,
            index=0,  # Default to latest quarter (already sorted newest first)
            help="Choose the quarter to analyze"
        )
    else:
        st.error(f"No quarterly data found for {selected_ticker_display}")
        selected_quarter = None

# Show comprehensive financial metrics display
if selected_ticker and selected_quarter:
    try:
        # Load data ONLY for selected ticker and quarter (with lookback)
        ticker_data = load_ticker_data(selected_ticker, selected_quarter)

        if ticker_data.empty:
            st.error(f"No financial data found for {selected_ticker_display} - {selected_quarter}")
            st.stop()

        # Step 1: Filter ticker data (already filtered, just format)
        ticker_data, pivot_data = filter_ticker_data(ticker_data, selected_ticker)

        # Step 2: Calculate financial metrics and growth rates
        calculated_metrics = calculate_financial_metrics(ticker_data, selected_quarter, selected_ticker)

        # Step 3: Create analysis tables (now returns TWO DataFrames)
        try:
            df_income_statement, df_balance_sheet = create_analysis_table(ticker_data, calculated_metrics, selected_quarter)
        except Exception as e:
            st.error(f"Error in create_analysis_table: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            df_income_statement = pd.DataFrame()
            df_balance_sheet = pd.DataFrame()

        # Step 4: Create summary tables with market share, prop book, and investment composition data
        market_share_table, prop_holdings_table, investment_composition_table = create_summary_tables(selected_ticker, selected_quarter, ticker_data)

        # Display the comprehensive financial tables
        if not calculated_metrics.empty and not df_income_statement.empty and not df_balance_sheet.empty:
            # Display Financial Metrics with tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📊 Comprehensive Metrics", "📈 Market Share", "🏢 Sector Comparison"])

            with tab1:
                st.subheader(f"Financial Metrics")
                st.markdown("*Units: VNDbn unless otherwise noted*")

                # Format the dataframe for display
                def format_value(val, metric_name):
                    """Format numeric values for display based on metric type"""
                    if pd.isna(val) or val == 0:
                        return "-"
                    elif isinstance(val, (int, float)):
                        # Special formatting for different metric types
                        if metric_name == 'Net Brokerage Fee':
                            # Net Brokerage Fee is in bps
                            return f"{val:.2f} bps"
                        elif metric_name in ['Brokerage Market Share', 'Margin/Equity %',
                                            'Margin Lending Rate', 'Margin Lending Spread',
                                            'CIR', 'Interest Rate', 'ROE']:
                            # Percentage metrics - add % symbol
                            return f"{val:.2f}%"
                        elif metric_name in ['Market Liquidity (Avg Daily)', 'Trading Value']:
                            # Already in billions
                            return f"{val:,.1f}"
                        else:
                            # Convert to billions for all other financial metrics (VND values)
                            val_billions = val / 1_000_000_000
                            return f"{val_billions:,.1f}"
                    return val

                # Display Income Statement Table
                st.markdown("### Income Statement")
                if not df_income_statement.empty:
                    # Apply formatting to numeric columns (skip 'Metric' column and growth columns)
                    display_income = df_income_statement.copy()
                    for col in display_income.columns:
                        if col not in ['Metric', 'QoQ Growth %', 'YoY Growth %']:
                            # Apply formatting with metric name context
                            display_income[col] = [
                                format_value(val, metric)
                                for val, metric in zip(display_income[col], display_income['Metric'])
                            ]

                    # Format growth columns with +/- sign and % symbol
                    if 'QoQ Growth %' in display_income.columns:
                        display_income['QoQ Growth %'] = display_income['QoQ Growth %'].apply(
                            lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x
                        )
                    if 'YoY Growth %' in display_income.columns:
                        display_income['YoY Growth %'] = display_income['YoY Growth %'].apply(
                            lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x
                        )

                    st.dataframe(display_income, use_container_width=True, hide_index=True)

                # Display Balance Sheet Table
                st.markdown("---")
                st.markdown("### Balance Sheet")
                if not df_balance_sheet.empty:
                    # Apply formatting to numeric columns (skip 'Metric' column and growth columns)
                    display_balance = df_balance_sheet.copy()
                    for col in display_balance.columns:
                        if col not in ['Metric', 'QoQ Growth %', 'YoY Growth %']:
                            # Apply formatting with metric name context
                            display_balance[col] = [
                                format_value(val, metric)
                                for val, metric in zip(display_balance[col], display_balance['Metric'])
                            ]

                    # Format growth columns with +/- sign and % symbol
                    if 'QoQ Growth %' in display_balance.columns:
                        display_balance['QoQ Growth %'] = display_balance['QoQ Growth %'].apply(
                            lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x
                        )
                    if 'YoY Growth %' in display_balance.columns:
                        display_balance['YoY Growth %'] = display_balance['YoY Growth %'].apply(
                            lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x
                        )

                    st.dataframe(display_balance, use_container_width=True, hide_index=True)

                # Add Proprietary Holdings after the metrics table
                st.markdown("---")

                st.markdown(f"#### Proprietary Holdings")
                if not prop_holdings_table.empty:
                    # Format the prop holdings table for display (values are already in billions)
                    prop_display = prop_holdings_table.copy()
                    for col_name in prop_display.columns:
                        prop_display[col_name] = prop_display[col_name].apply(
                            lambda x: f"{x:,.1f}" if x > 0 else "-"
                        )
                    st.dataframe(prop_display, use_container_width=True)

                    # Add expandable volume table
                    with st.expander("📊 View Holdings Volume (shares)"):
                        # Get quarters from prop holdings table
                        quarters_for_volume = prop_holdings_table.columns.tolist()

                        # Calculate volumes
                        volume_table = calculate_prop_holdings_volume(selected_ticker, quarters_for_volume)

                        if not volume_table.empty:
                            # Format volumes as integers with thousand separators
                            volume_display = volume_table.copy()
                            for col in volume_display.columns:
                                volume_display[col] = volume_display[col].apply(
                                    lambda x: f"{int(x):,}" if x > 0 else "-"
                                )
                            st.dataframe(volume_display, use_container_width=True)
                            st.caption("*Volume calculated as Market Value (billions VND) / Quarter-End Closing Price*")
                        else:
                            st.info("Volume data not available")
                else:
                    st.info(f"No proprietary holdings data for {selected_ticker_display}")

            with tab2:
                st.subheader("Market Share & Trading Activity")

                # Display market share table
                if not market_share_table.empty:
                    st.markdown("#### Market Share Evolution")
                    st.dataframe(market_share_table, use_container_width=True, hide_index=True)
                else:
                    st.info(f"Market share data not available for {selected_ticker_display}")

            with tab3:
                st.subheader(f"Cross-Broker Comparison")
                st.markdown("*Compare financial metrics across multiple brokers for the selected quarter*")

                # Broker and quarter selection side by side
                col_broker, col_quarter = st.columns(2)

                with col_broker:
                    st.markdown("#### Select Brokers to Compare")

                    # Create broker list ordered by tier (same as main page)
                    comparison_ordered_tickers = []
                    for group_name, tickers in broker_groups.items():
                        for ticker in tickers:
                            if ticker in available_tickers and ticker in DISPLAY_TICKERS_WHITELIST:
                                comparison_ordered_tickers.append(ticker)

                    # Add any brokers not in groups at the end (only if whitelisted)
                    ungrouped = [t for t in available_tickers if not any(t in group for group in broker_groups.values()) and t in DISPLAY_TICKERS_WHITELIST]
                    comparison_ordered_tickers.extend(ungrouped)

                    # Create display names mapping
                    ticker_display_map = {ticker: get_display_ticker(ticker) for ticker in comparison_ordered_tickers}

                    # Multiselect for brokers
                    selected_brokers = st.multiselect(
                        "Choose brokers:",
                        options=comparison_ordered_tickers,
                        format_func=lambda x: ticker_display_map[x],
                        default=[selected_ticker],
                        help="Select one or more brokers to compare"
                    )

                with col_quarter:
                    st.markdown("#### Select Quarter")
                    # Use the same quarter list as the main page
                    comparison_quarter = st.selectbox(
                        "Choose quarter:",
                        ticker_quarters,
                        index=ticker_quarters.index(selected_quarter) if selected_quarter in ticker_quarters else 0,
                        help="Choose the quarter for comparison",
                        key="comparison_quarter_selector"
                    )

                # Metric selection
                st.markdown("---")
                st.markdown("#### Select Metrics to Compare")

                # Available metrics (same as Charts page)
                available_metrics = {
                    'PBT': 'PBT',
                    'NPAT': 'NPAT',
                    'ROE': 'ROE',
                    'Margin Balance': 'Margin_Lending_book',
                    'Margin Lending Rate': 'MARGIN_LENDING_RATE',
                    'Brokerage Market Share': 'Brokerage Market Share',
                    'Net Brokerage Income': 'Net_Brokerage_Income',
                    'Margin Income': 'Net_Margin_lending_Income',
                    'Investment Income': 'Net_investment_income',
                    'Total Operating Income': 'Total_Operating_Income',
                    'CIR': 'CIR',
                    'Interest Rate': 'INTEREST_RATE',
                    'Total Debt': 'Total_Debt_Balance',
                }

                # Default metrics
                default_metrics = ['PBT', 'ROE', 'Margin Balance', 'Margin Lending Rate', 'Brokerage Market Share']

                selected_metric_names = st.multiselect(
                    "Choose metrics:",
                    options=list(available_metrics.keys()),
                    default=default_metrics,
                    help="Select one or more metrics to compare across brokers"
                )

                # Generate comparison if brokers and metrics are selected
                if selected_brokers and selected_metric_names:
                    st.markdown("---")

                    # Load data for all selected brokers
                    comparison_data = []

                    for ticker in selected_brokers:
                        ticker_comparison_data = load_ticker_data(ticker, comparison_quarter)

                        if not ticker_comparison_data.empty:
                            # Parse quarter
                            try:
                                quarter_num = int(comparison_quarter[0])
                                year_str = comparison_quarter[-2:]
                                year = 2000 + int(year_str) if int(year_str) < 50 else 1900 + int(year_str)
                            except:
                                continue

                            broker_metrics = {'Broker': get_display_ticker(ticker)}

                            for metric_name in selected_metric_names:
                                metric_code = available_metrics[metric_name]

                                # Special handling for Brokerage Market Share
                                if metric_name == 'Brokerage Market Share':
                                    # Try HSX API first
                                    hsx_data = fetch_market_share(ticker, comparison_quarter)
                                    if hsx_data['market_share'] > 0:
                                        broker_metrics[metric_name] = hsx_data['market_share']
                                    else:
                                        # Calculate fallback
                                        institution_shares = get_calc_metric_value(ticker_comparison_data, ticker, year, quarter_num, 'Institution_shares_trading_value')
                                        investor_shares = get_calc_metric_value(ticker_comparison_data, ticker, year, quarter_num, 'Investor_shares_trading_value')
                                        total_trading_value = institution_shares + investor_shares

                                        market_liquidity_df = load_market_liquidity_data()
                                        if not market_liquidity_df.empty:
                                            liquidity_row = market_liquidity_df[
                                                (market_liquidity_df['Year'] == year) &
                                                (market_liquidity_df['Quarter'] == quarter_num)
                                            ]
                                            if not liquidity_row.empty:
                                                avg_daily_turnover_bn = liquidity_row.iloc[0]['Avg Daily Turnover (B VND)']
                                                trading_days = liquidity_row.iloc[0]['Trading Days']
                                                total_market_value = avg_daily_turnover_bn * 1_000_000_000 * trading_days

                                                if total_market_value and total_market_value != 0 and total_trading_value > 0:
                                                    market_share = (total_trading_value / total_market_value) / 2 * 100
                                                    broker_metrics[metric_name] = market_share
                                                else:
                                                    broker_metrics[metric_name] = 0
                                            else:
                                                broker_metrics[metric_name] = 0
                                        else:
                                            broker_metrics[metric_name] = 0
                                else:
                                    # Regular metric
                                    value = get_calc_metric_value(ticker_comparison_data, ticker, year, quarter_num, metric_code)
                                    broker_metrics[metric_name] = value

                            comparison_data.append(broker_metrics)

                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)

                        # Transpose: Metrics as rows, Brokers as columns
                        # Set Broker as index then transpose
                        comparison_df_transposed = comparison_df.set_index('Broker').T

                        # Reset index to have Metric as a column
                        comparison_df_transposed = comparison_df_transposed.reset_index()
                        comparison_df_transposed.rename(columns={'index': 'Metric'}, inplace=True)

                        # Display comparison table
                        st.markdown(f"### Comparison Table")
                        st.markdown("*Rows: Metrics | Columns: Brokers*")

                        # Format the display - metrics as rows, brokers as columns
                        display_comparison = comparison_df_transposed.copy()

                        # Format each broker column based on the metric
                        for col in display_comparison.columns:
                            if col != 'Metric':  # Skip the Metric column
                                formatted_values = []
                                for idx, row in display_comparison.iterrows():
                                    metric_name = row['Metric']
                                    value = row[col]

                                    if pd.isna(value) or value == 0:
                                        formatted_values.append("-")
                                    elif metric_name in ['ROE', 'Margin Lending Rate', 'Brokerage Market Share', 'CIR', 'Interest Rate']:
                                        # Percentage metrics
                                        formatted_values.append(f"{value:.2f}%")
                                    else:
                                        # Value metrics (convert to billions)
                                        formatted_values.append(f"{value/1_000_000_000:,.1f}")

                                display_comparison[col] = formatted_values

                        st.dataframe(display_comparison, use_container_width=True, hide_index=True)

                    else:
                        st.warning("No data available for selected brokers")

                elif not selected_brokers:
                    st.info("👆 Please select at least one broker to compare")
                elif not selected_metric_names:
                    st.info("👆 Please select at least one metric to compare")

        else:
            st.warning(f"Could not calculate metrics for {selected_ticker_display} in {selected_quarter}")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

st.markdown("---")
