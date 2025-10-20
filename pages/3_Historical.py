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
    """Step 3: Combine historical data and calculated metrics for analysis - Last 6 Quarters"""

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

    # Display metrics we want to show (excluding investment book items - they're displayed separately)
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

        # Sort rows by the most recent quarter value (descending)
        if len(sorted_quarters) > 0:
            most_recent_quarter = sorted_quarters[-1]
            holdings_pivot = holdings_pivot.sort_values(by=most_recent_quarter, ascending=False)

        return holdings_pivot

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
    # Create grouped broker options
    broker_options = []
    broker_mapping = {}  # Map display string to actual ticker code

    for group_name, tickers in broker_groups.items():
        # Add group header
        broker_options.append(f"──── {group_name} ────")
        broker_mapping[f"──── {group_name} ────"] = None  # Header has no ticker

        # Add brokers in this group
        for ticker in tickers:
            if ticker in available_tickers:
                display_name = get_display_ticker(ticker)
                broker_options.append(f"  {display_name}")
                broker_mapping[f"  {display_name}"] = ticker

    # Add any brokers not in groups
    ungrouped = [t for t in available_tickers if not any(t in group for group in broker_groups.values())]
    if ungrouped:
        broker_options.append("──── Others ────")
        broker_mapping["──── Others ────"] = None
        for ticker in ungrouped:
            display_name = get_display_ticker(ticker)
            broker_options.append(f"  {display_name}")
            broker_mapping[f"  {display_name}"] = ticker

    # Find default selection (SSI)
    default_ticker = 'SSI'
    default_display = f"  {get_display_ticker(default_ticker)}"
    default_index = broker_options.index(default_display) if default_display in broker_options else 0

    # Skip headers in selection
    while default_index < len(broker_options) and broker_mapping[broker_options[default_index]] is None:
        default_index += 1

    selected_broker_display = st.selectbox(
        "Select Broker:",
        options=broker_options,
        index=default_index,
        help="Choose a broker to analyze"
    )

    # Get the actual ticker code
    selected_ticker = broker_mapping.get(selected_broker_display)

    # If a header was selected, default to first valid broker
    if selected_ticker is None:
        for opt in broker_options:
            if broker_mapping[opt] is not None:
                selected_ticker = broker_mapping[opt]
                break

    # Get the display name for UI
    selected_ticker_display = get_display_ticker(selected_ticker) if selected_ticker else "Unknown"

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

        # Step 3: Create analysis table
        try:
            analysis_table = create_analysis_table(ticker_data, calculated_metrics, selected_quarter)
        except Exception as e:
            st.error(f"Error in create_analysis_table: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            analysis_table = pd.DataFrame()

        # Step 4: Create summary tables with market share, prop book, and investment composition data
        market_share_table, prop_holdings_table, investment_composition_table = create_summary_tables(selected_ticker, selected_quarter, ticker_data)

        # Display the comprehensive financial tables
        if not calculated_metrics.empty and not analysis_table.empty:
            # Display Financial Metrics with tabs for better organization
            tab1, tab2 = st.tabs(["Comprehensive Metrics", "Market Share"])

            with tab1:
                st.subheader(f"Financial Metrics")
                st.markdown("*Units: VNDbn unless otherwise noted*")

                # Display the full analysis table
                if not analysis_table.empty:
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

                    # Apply formatting to numeric columns (skip 'Metric' column and growth columns)
                    display_df = analysis_table.copy()
                    for col in display_df.columns:
                        if col not in ['Metric', 'QoQ Growth %', 'YoY Growth %']:
                            # Apply formatting with metric name context
                            display_df[col] = [
                                format_value(val, metric)
                                for val, metric in zip(display_df[col], display_df['Metric'])
                            ]

                    # Format growth columns with +/- sign and % symbol
                    if 'QoQ Growth %' in display_df.columns:
                        display_df['QoQ Growth %'] = display_df['QoQ Growth %'].apply(
                            lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x
                        )
                    if 'YoY Growth %' in display_df.columns:
                        display_df['YoY Growth %'] = display_df['YoY Growth %'].apply(
                            lambda x: f"{x:+.1f}%" if isinstance(x, (int, float)) else x
                        )

                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                # Add Investment Book and Prop Holdings after the metrics table
                st.markdown("---")

                # Investment Book Composition and Proprietary Holdings side by side
                col1, col2 = st.columns(2)

                with col1:
                    num_quarters = len(investment_composition_table.columns) - 1 if not investment_composition_table.empty else 6  # Subtract 1 for 'Investment Type' column
                    st.markdown(f"#### Investment Book Composition")
                    if not investment_composition_table.empty:
                        st.dataframe(investment_composition_table, use_container_width=True, hide_index=True)
                    else:
                        st.info(f"No investment holdings data for {selected_ticker_display}")

                with col2:
                    st.markdown(f"#### Proprietary Holdings")
                    if not prop_holdings_table.empty:
                        # Format the prop holdings table for display (values are already in billions)
                        prop_display = prop_holdings_table.copy()
                        for col_name in prop_display.columns:
                            prop_display[col_name] = prop_display[col_name].apply(
                                lambda x: f"{x:,.1f}" if x > 0 else "-"
                            )
                        st.dataframe(prop_display, use_container_width=True)
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

        else:
            st.warning(f"Could not calculate metrics for {selected_ticker_display} in {selected_quarter}")

    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

st.markdown("---")
