import streamlit as st
import toml

# Load theme from config.toml
theme_config = toml.load("utils/config.toml")
theme = theme_config["theme"]
primary_color = theme["primaryColor"]
background_color = theme["backgroundColor"]
secondary_background_color = theme["secondaryBackgroundColor"]
text_color = theme["textColor"]
font_family = theme["font"]
import pandas as pd
import sys
import os
from utils.keycode_matcher import load_keycode_map, match_keycodes
from utils.data import calculate_income_statement_items, calculate_market_turnover_and_trading_days

st.set_page_config(page_title="Forecast", layout="wide")

if st.sidebar.button("Reload Data"):
    st.cache_data.clear()

@st.cache_data
def load_data():
    try:
        df_index = pd.read_csv("sql/INDEX.csv")
        # Extract 'Year' from 'TRADINGDATE' for df_index
        if 'TRADINGDATE' in df_index.columns:
            df_index['Year'] = pd.to_datetime(df_index['TRADINGDATE']).dt.year
        keycode_map = load_keycode_map('sql/IRIS_KEYCODE.csv')
        df_is = match_keycodes('sql/IS_security.csv', keycode_map)
        df_bs = match_keycodes('sql/BS_security.csv', keycode_map)
        df_note = match_keycodes('sql/Note_security.csv', keycode_map)

        # Extract 'Year' from 'YEARREPORT' for splitting
        for df in [df_is, df_bs, df_note]:
            if 'YEARREPORT' in df.columns:
                df['Year'] = df['YEARREPORT']
        df_forecast = pd.read_csv("sql/FORECAST.csv")
        df_turnover = pd.read_excel("sql/turnover.xlsx")
        return df_index, df_bs, df_is, df_note, df_forecast, df_turnover
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

df_index, df_bs, df_is, df_note, df_forecast, df_turnover = load_data()

from datetime import datetime
current_year = datetime.now().year

def split_historical_forecast(df, year_col='Year', current_year=current_year):
    historical = df[df[year_col] < current_year]
    forecast = df[df[year_col] >= current_year]
    return historical, forecast


historical_is, forecast_is = split_historical_forecast(df_is)
historical_bs, forecast_bs = split_historical_forecast(df_bs)
historical_note, forecast_note = split_historical_forecast(df_note)

if not df_index.empty:
    # Filter to use only COMGROUPCODE == 'VNINDEX'
    df_index = df_index[df_index['COMGROUPCODE'] == 'VNINDEX']
    brokers = ['SSI', 'HCM', 'VND', 'VCI', 'MBS', 'SHS', 'BSI', 'VIX']
    df_forecast = df_forecast[df_forecast['TICKER'].isin(brokers)]

    # Sidebar controls
    selected_broker = st.sidebar.selectbox(
        "Select Broker:",
        options=brokers,
        index=0
    )
    
    # Add 2026 forecast button
    show_2026 = st.sidebar.button("Include 2026 Forecast", key="include_2026")
    
    # Store 2026 state in session state
    if 'show_2026_forecast' not in st.session_state:
        st.session_state.show_2026_forecast = False
    
    if show_2026:
        st.session_state.show_2026_forecast = not st.session_state.show_2026_forecast

    # Filter for selected broker and years 2020-2025/2026
    max_year = 2026 if st.session_state.show_2026_forecast else 2025
    df_broker_hist = df_forecast[(df_forecast['TICKER'] == selected_broker) & (df_forecast['DATE'] >= 2020) & (df_forecast['DATE'] < 2025)]
    df_broker_forecast = df_forecast[(df_forecast['TICKER'] == selected_broker) & (df_forecast['DATE'] == 2025)]
    df_broker_forecast_2026 = df_forecast[(df_forecast['TICKER'] == selected_broker) & (df_forecast['DATE'] == 2026)]

    # Example: Add your custom calculations below (filtered by selected broker)

    def filter_full_year(df):
        # Assumes STARTDATE and ENDDATE are in datetime or string format 'YYYY-MM-DD'
        df = df.copy()
        df['STARTDATE'] = pd.to_datetime(df['STARTDATE'], errors='coerce')
        df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')
        df['Year'] = df['Year'].astype(int)
        mask = (
            (df['STARTDATE'].dt.month == 1) & (df['STARTDATE'].dt.day == 1) &
            (df['ENDDATE'].dt.month == 12) & (df['ENDDATE'].dt.day == 31)
        )
        return df[mask]

    filtered_is = filter_full_year(historical_is[historical_is['TICKER'] == selected_broker])
    filtered_bs = filter_full_year(historical_bs[historical_bs['TICKER'] == selected_broker])

    # Calculate income statement items using the utility function
    income_statement_items = calculate_income_statement_items(filtered_is, filtered_bs)
    
    # Extract individual items for backward compatibility
    fx_gain_loss_by_year = income_statement_items['fx_gain_loss_by_year']
    affiliates = income_statement_items['affiliates']
    associates = income_statement_items['associates']
    deposit_inc = income_statement_items['deposit_inc']
    interest_exp = income_statement_items['interest_exp']
    ib_inc = income_statement_items['ib_inc']
    net_brokerage = income_statement_items['net_brokerage']
    trading_inc = income_statement_items['trading_inc']
    interest_inc = income_statement_items['interest_inc']
    inv_inc = income_statement_items['inv_inc']
    margin_inc = income_statement_items['margin_inc']
    other_inc = income_statement_items['other_inc']
    other_op_inc = income_statement_items['other_op_inc']
    fee_inc = income_statement_items['fee_inc']
    capital_inc = income_statement_items['capital_inc']
    total_operating_income = income_statement_items['total_operating_income']
    borrowing_balance = income_statement_items['borrowing_balance']
    sga = income_statement_items['sga']
    pbt = income_statement_items['pbt']
    NPAT = income_statement_items['NPAT']
    margin_balance = income_statement_items['margin_balance']
    # Add more custom calculations and manipulations as needed

    # Calculate average daily market turnover and trading days using the utility function
    market_turnover_by_year, trading_days_2025 = calculate_market_turnover_and_trading_days(df_index, current_year=2024)

    # Calculate defaults: market share from 2024 actual, others from 2025 forecast
    def get_forecast_value(keycodename):
        d = df_broker_forecast[df_broker_forecast['KEYCODENAME'] == keycodename]
        return d['VALUE'].sum() if not d.empty else 0
    
    # Market share from 2024 actual data
    turnover_2024 = df_turnover[(df_turnover['Year'] == 2024) & (df_turnover['Ticker'] == selected_broker)]
    market_share_2024_default = (turnover_2024['Company turnover'].iloc[0] / turnover_2024['Market turnover'].iloc[0] / 2) * 100
    
    # Others from 2025 forecast
    market_turnover_daily_2024_default = market_turnover_by_year[market_turnover_by_year['Year'] == 2024]['avg_daily_turnover'].iloc[0] / 1e9
    net_fee_2024_default = net_brokerage[net_brokerage['Year'] == 2024]['Net Brokerage Income'].iloc[0] / (turnover_2024['Company turnover'].iloc[0] * 1e9)
    margin_2024_default = get_forecast_value('Margin lending book') / 1e9
    
    # CIR calculation with safety check - use baseline forecast for 2025 default
    sga_value = abs(get_forecast_value('SG&A'))
    total_op_income = get_forecast_value('Total Operating Income')
    net_inv_income = get_forecast_value('Net Investment Income')
    denominator = total_op_income - net_inv_income
    cir_2025_default = sga_value / denominator if denominator > 0 else 0.0
    
    # For display purposes, convert to percentage
    cir_2025_default_percent = cir_2025_default * 100

    # Initialize default values (will be updated by controls in respective segments)
    market_share_adj = market_share_2024_default
    market_turnover_daily_adj = market_turnover_daily_2024_default
    net_fee_adj = net_fee_2024_default * 100
    margin_adj = margin_2024_default
    cir_adj = cir_2025_default_percent
    
    # Initialize 2026 visibility state
    show_2026 = False

    # For 2025, use adjustment values to set market share and market turnover
    market_share_2025 = market_share_adj / 100.0
    market_turnover_2025 = market_turnover_daily_adj * trading_days_2025  # Convert from bn to actual value
    company_turnover_2025 = market_share_2025 * market_turnover_2025 * 2
    net_fee_2025 = net_fee_adj / 100.0  # Convert back to decimal for calculations
    cir_2025 = cir_adj / 100.0
    
    # Initialize 2026 values with same defaults as 2025
    market_share_2026 = market_share_2025
    market_turnover_2026 = market_turnover_2025  # Same as 2025 by default
    company_turnover_2026 = market_share_2026 * market_turnover_2026 * 2
    net_fee_2026 = net_fee_2025
    cir_2026 = cir_2025
    margin_2026 = margin_adj

    # You can add more calculation functions below and use more sliders for other metrics

    # Ensure TRADINGDATE is datetime
    df_index['TRADINGDATE'] = pd.to_datetime(df_index['TRADINGDATE'])

    # Create annual and quarter columns
    df_index['Annual'] = df_index['TRADINGDATE'].dt.to_period('Y').astype(str)
    df_index['Quarter'] = df_index['TRADINGDATE'].dt.to_period('Q').astype(str)

    # Calculate net foreign trading value (buy - sell, matched)
    df_index['Net_Foreign_Trading'] = df_index['FOREIGNBUYVALUEMATCHED'] - df_index['FOREIGNSELLVALUEMATCHED']

    # Group by month and calculate total and average
    summary_annual = df_index.groupby('Annual').agg(
        Total_Value=('TOTALVALUE', 'sum'),
        Avg_Daily_Value=('TOTALVALUE', 'mean'),
        Total_Net_Foreign_Trading=('Net_Foreign_Trading', 'sum'),
        Avg_Daily_Net_Foreign_Trading=('Net_Foreign_Trading', 'mean'),
        Days=('TRADINGDATE', lambda x: x.nunique())
    ).reset_index()

    summary_annual = summary_annual.sort_values('Annual')

    # Format numbers into billions with thousand separators
    for col in ['Total_Value', 'Avg_Daily_Value', 'Total_Net_Foreign_Trading', 'Avg_Daily_Net_Foreign_Trading']:
        summary_annual[col] = summary_annual[col].apply(lambda x: f"{x:,.0f}")

    # Group by quarter and calculate total and average
    summary_quarter = df_index.groupby('Quarter').agg(
        Total_Value=('TOTALVALUE', 'sum'),
        Avg_Daily_Value=('TOTALVALUE', 'mean'),
        Total_Net_Foreign_Trading=('Net_Foreign_Trading', 'sum'),
        Avg_Daily_Net_Foreign_Trading=('Net_Foreign_Trading', 'mean'),
        Days=('TRADINGDATE', lambda x: x.nunique())
    ).reset_index()

    summary_quarter = summary_quarter.sort_values('Quarter')

    # Format numbers into billions with thousand separators
    for col in ['Total_Value', 'Avg_Daily_Value', 'Total_Net_Foreign_Trading', 'Avg_Daily_Net_Foreign_Trading']:
        summary_quarter[col] = summary_quarter[col].apply(lambda x: f"{x/1e9:,.0f}")


    # Universal function to build financial statement items
    def build_financial_statement(df, items):
        result = pd.DataFrame({'DATE': df['DATE'].unique()})
        for item_name, filter_func in items.items():
            if item_name == 'Net Fees Income':
                # Sum Net Brokerage Income and Net IB Income for each date
                brokerage_func = items['Net Brokerage Income']
                ib_func = items['Net IB Income']
                result[item_name] = result['DATE'].apply(
                    lambda d: brokerage_func(df[df['DATE'] == d]) + ib_func(df[df['DATE'] == d])
                )
            elif item_name == 'Net Capital Income':
                # Example: sum Net Investment Income and Margin Lending Income
                invest_func = items.get('Net Investment Income', lambda d: 0)
                margin_func = items.get('Margin Lending Income', lambda d: 0)
                result[item_name] = result['DATE'].apply(
                    lambda d: invest_func(df[df['DATE'] == d]) + margin_func(df[df['DATE'] == d])
                )
            else:
                result[item_name] = result['DATE'].apply(lambda d: filter_func(df[df['DATE'] == d]))
        return result

    # Example: Add items as you go
    items = {
        'Net Brokerage Income': lambda d: d['Net Brokerage Income'].values[0] if 'Net Brokerage Income' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Net Brokerage Income', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Market Turnover': None,  # Will be calculated in the function
        'Company Turnover': None,  # Will be calculated in the function
        'Brokerage Market Share': None,  # Will be calculated in the function
        'Net Brokerage Fee': None, # Will be calculated in the function
        'Net IB Income': None,  # Will be calculated in the function
        'Net other operating income': lambda d: d['Net other operating income'].values[0] if 'Net other operating income' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Net other operating income', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Net Fees Income': None,  # Will be calculated in the function
        'Net Investment Income': None,  # Will be calculated in the function
        'Margin Lending Income': None,  # Will be calculated in the function
        'Margin Balance': lambda d: d['Margin Balance'].values[0] if 'Margin Balance' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Margin lending book', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Lending rate': None, # Will be calculated in the function
        'Capital Income': None,  # Will be calculated in the function
        'Total Operating Income': None,  # Will be calculated in the function
        'SG&A': None,  # Will be calculated in the function
        'FX gain/loss': lambda d: d['FX gain/loss'].values[0] if 'FX gain/loss' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'FX gain/(loss)', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Deposit income': lambda d: d['Deposit income'].values[0] if 'Deposit income' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Deposit income', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Gain/loss from affiliates divestment': lambda d: d['Gain/loss from affiliates divestment'].values[0] if 'Gain/loss from affiliates divestment' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Gain/(loss) in affliates divestments', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Income from associate companies': lambda d: d['Income from associate companies'].values[0] if 'Income from associate companies' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Income from associate companies', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Net other income': lambda d: d['Net other income'].values[0] if 'Net other income' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Net other income', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Other incomes': None,  # Will be calculated in the function
        'Interest expense': lambda d: d['Interest expense'].values[0] if 'Interest expense' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Interest Expense', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'Borrowing balance': lambda d: d['Borrowing balance'].values[0] if 'Borrowing balance' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Borrowing balance', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'CIR': None,  # Will be calculated in the function
        'PBT': None,  # Will be calculated in the function
        'Tax expense': lambda d: d['Tax expense'].values[0] if 'Tax expense' in d.columns and not d.empty else (d.loc[d['KEYCODENAME'] == 'Tax expense', 'VALUE'].sum() if 'KEYCODENAME' in d.columns else 0),
        'NPAT': None, # Will be calculated in the function
        # Add more items as needed
    }

    def get_value(df, year, col, keycodename=None):
        # For forecast years, use KEYCODENAME/VALUE from FORECAST.csv
        if year in [2025, 2026] and keycodename is not None:
            d = df[(df['DATE'] == year) & (df['KEYCODENAME'] == keycodename)]
            return d['VALUE'].sum() if not d.empty else 0
        else:
            d = df[df['DATE'] == year]
            return d[col].values[0] if col in d.columns and not d.empty else 0

    # Universal function to build financial statement items (pivoted)
    def build_financial_statement_pivot(df, items, cir_adj_param=None):
        years = sorted(df['DATE'].unique())
        
        # Use the passed parameter or fall back to global variable
        cir_adj_to_use = cir_adj_param if cir_adj_param is not None else cir_adj
        
        # Calculate baseline lending rate for consistency
        def get_value_baseline(df, year, col, keycodename=None):
            if year == 2025 and keycodename is not None:
                d = df[(df['DATE'] == year) & (df['KEYCODENAME'] == keycodename)]
                return d['VALUE'].sum() if not d.empty else 0
            else:
                d = df[df['DATE'] == year]
                return d[col].values[0] if col in d.columns and not d.empty else 0
        
        baseline_margin_income_2025 = get_value_baseline(df_broker_forecast, 2025, 'Margin Lending Income', keycodename='Net Margin lending Income')
        baseline_margin_balance_2025 = get_value_baseline(df_broker_forecast, 2025, 'Margin Balance', keycodename='Margin lending book')
        margin_balance_2024 = margin_balance[margin_balance['Year'] == 2024]['Margin Balance'].iloc[0] if not margin_balance[margin_balance['Year'] == 2024].empty else 0
        baseline_avg_balance = (margin_balance_2024 + baseline_margin_balance_2025) / 2
        
        if baseline_avg_balance != 0:
            baseline_lending_rate = baseline_margin_income_2025 / baseline_avg_balance
        else:
            baseline_lending_rate = 0.15
        
        data = {}
        for item_name, filter_func in items.items():
            if item_name == 'Market Turnover':
                # Use INDEX file for historical, sidebar for forecast years
                values = []
                for year in years:
                    if year == 2025:
                        values.append(market_turnover_2025)
                    elif year == 2026:
                        values.append(market_turnover_2026)
                    else:
                        mt_row = market_turnover_by_year[market_turnover_by_year['Year'] == year]
                        if not mt_row.empty:
                            avg_daily = mt_row['avg_daily_turnover'].iloc[0]
                            trading_days = mt_row['trading_days'].iloc[0]
                            annual_turnover = avg_daily * trading_days / 1e9
                            values.append(annual_turnover)
                        else:
                            values.append('')
                data[item_name] = values
            elif item_name == 'Net Brokerage Income':
                values = []
                for year in years:
                    if year == 2025:
                        # Use sidebar-adjusted formula for 2025
                        values.append(net_fee_2025 * company_turnover_2025 * 1e9)
                    elif year == 2026:
                        # Use sidebar-adjusted formula for 2026
                        values.append(net_fee_2026 * company_turnover_2026 * 1e9)
                    else:
                        # Use column value for historical years if present
                        row = df[df['DATE'] == year]
                        if 'Net Brokerage Income' in row.columns and not row.empty:
                            val = row['Net Brokerage Income'].values[0]
                        else:
                            # fallback to forecast structure if needed
                            val = row.loc[row['KEYCODENAME'] == 'Net Brokerage Income', 'VALUE'].sum() if 'KEYCODENAME' in row.columns else 0
                        values.append(val)
                data[item_name] = values
            elif item_name == 'Company Turnover':
                broker_col = 'Ticker'
                company_col = 'Company turnover'
                values = []
                debug_rows = []
                for year in years:
                    if year == 2025:
                        values.append(company_turnover_2025)
                        debug_rows.append({'Year': year, 'Value': company_turnover_2025, 'Source': 'Sidebar'})
                    elif year == 2026:
                        values.append(company_turnover_2026)
                        debug_rows.append({'Year': year, 'Value': company_turnover_2026, 'Source': 'Sidebar'})
                    else:
                        row = df_turnover[(df_turnover['Year'] == year) & (df_turnover[broker_col] == selected_broker)]
                        if not row.empty and company_col in row.columns:
                            val = row.iloc[0][company_col]
                        else:
                            val = ''
                        values.append(val)
                        debug_rows.append({'Year': year, 'Value': val, 'Source': 'Excel'})
                data[item_name] = values
            elif item_name == 'Brokerage Market Share':
                broker_col = 'Ticker'
                company_col = 'Company turnover'
                market_col = 'Market turnover'
                shares = []
                for year in years:
                    if year == 2025:
                        shares.append(market_share_2025)
                    elif year == 2026:
                        shares.append(market_share_2026)
                    else:
                        row = df_turnover[(df_turnover['Year'] == year) & (df_turnover[broker_col] == selected_broker)]
                        company_turnover = row[company_col].values[0] if not row.empty else 0
                        market_turnover = row[market_col].values[0] if not row.empty else 0
                        if market_turnover == 0:
                            share = 0
                        else:
                            share = company_turnover / market_turnover / 2
                        shares.append(share)
                data[item_name] = shares
            elif item_name == 'Net Brokerage Fee':
                fee = []
                for i, year in enumerate(years):
                    if year == 2025:
                        fee.append(net_fee_adj / 100)
                    elif year == 2026:
                        fee.append(net_fee_2026 if 'net_fee_2026' in locals() else net_fee_2025)
                    else:
                        nbi = data.get('Net Brokerage Income', [None]*len(years))[i]
                        ct = data.get('Company Turnover', [None]*len(years))[i]
                        try:
                            nbi_val = float(nbi)
                            ct_val = float(ct)
                            if ct_val != 0:
                                fee_val = nbi_val / ct_val / 1e9
                            else:
                                fee_val = None
                        except (TypeError, ValueError):
                            fee_val = None
                        fee.append(fee_val)
                data[item_name] = fee
            elif item_name == 'Net Fees Income':
                values = []
                for year in years:
                    if year == 2025:
                        # For forecast year, calculate from components using data dictionary
                        nbi = data.get('Net Brokerage Income', [0]*len(years))[years.index(year)]
                        ib = data.get('Net IB Income', [0]*len(years))[years.index(year)]
                        other_op = data.get('Net other operating income', [0]*len(years))[years.index(year)]
                        net_fees = nbi + ib + other_op
                    else:
                        # For historical years, calculate from historical components
                        nbi = get_value(df, year, 'Net Brokerage Income', keycodename=None)
                        ib = get_value(df, year, 'Net IB Income', keycodename=None)
                        other_op = get_value(df, year, 'Net other operating income', keycodename=None)
                        net_fees = nbi + ib + other_op
                    values.append(net_fees)
                data[item_name] = values
            elif item_name == 'Lending rate':
                values = []
                for i, year in enumerate(years):
                    if year == 2025:
                        # Use baseline lending rate for 2025 regardless of balance adjustments
                        values.append(baseline_lending_rate)
                    else:
                        # For historical years, calculate based on actual data
                        margin_income = data.get('Margin Lending Income', [0]*len(years))[i]
                        current_balance = data.get('Margin Balance', [0]*len(years))[i]
                        # Get previous year balance if available
                        if i > 0:
                            prev_balance = data.get('Margin Balance', [0]*len(years))[i-1]
                            avg_balance = (current_balance + prev_balance) / 2
                        else:
                            # For first year, use current balance only
                            avg_balance = current_balance
                        
                        # Calculate lending rate
                        if avg_balance != 0:
                            lending_rate = margin_income / avg_balance
                        else:
                            lending_rate = 0
                        
                        values.append(lending_rate)
                data[item_name] = values
            elif item_name == 'Margin Lending Income':
                values = []
                for i, year in enumerate(years):
                    if year == 2025:
                        margin_balance_2025 = margin_adj * 1e9
                        # Get 2024 margin balance from historical data, not from data dictionary
                        margin_balance_2024 = margin_balance[margin_balance['Year'] == 2024]['Margin Balance'].iloc[0] if not margin_balance[margin_balance['Year'] == 2024].empty else 0
                        avg_balance = (margin_balance_2024 + margin_balance_2025) / 2
                        margin_income_2025 = baseline_lending_rate * avg_balance
                        values.append(margin_income_2025)
                    elif year == 2026:
                        margin_balance_2026_val = margin_2026 * 1e9 if 'margin_2026' in locals() else margin_adj * 1e9
                        margin_balance_2025_val = margin_adj * 1e9
                        avg_balance_2026 = (margin_balance_2025_val + margin_balance_2026_val) / 2
                        margin_income_2026 = baseline_lending_rate * avg_balance_2026
                        values.append(margin_income_2026)
                    else:
                        values.append(get_value(df, year, 'Margin Lending Income', keycodename='Net Margin lending Income'))
                data[item_name] = values
            elif item_name == 'Net Investment Income':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Net Investment Income', keycodename='Net Investment Income'))
                data[item_name] = values
            elif item_name == 'Net IB Income':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Net IB Income', keycodename='Net IB Income'))
                data[item_name] = values
            elif item_name == 'Margin Balance':
                values = []
                for year in years:
                    if year == 2025:
                        values.append(margin_adj * 1e9)  # Convert from billions to actual units
                    elif year == 2026:
                        values.append(margin_2026 * 1e9 if 'margin_2026' in locals() else margin_adj * 1e9)
                    else:
                        values.append(get_value(df, year, 'Margin Balance', keycodename='Margin lending book'))
                data[item_name] = values
            elif item_name == 'FX gain/loss':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'FX gain/loss', keycodename='FX gain/(loss)'))
                data[item_name] = values
            elif item_name == 'Deposit income':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Deposit income', keycodename='Deposit income'))
                data[item_name] = values
            elif item_name == 'Gain/loss from affiliates divestment':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Gain/loss from affiliates divestment', keycodename='Gain/(loss) in affliates divestments'))
                data[item_name] = values
            elif item_name == 'Income from associate companies':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Income from associate companies', keycodename='Income from associate companies'))
                data[item_name] = values
            elif item_name == 'Net other income':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Net other income', keycodename='Net other income'))
                data[item_name] = values
            elif item_name == 'Other incomes':
                values = []
                for year in years:
                    fx = get_value(df, year, 'FX gain/loss', keycodename='FX gain/(loss)')
                    deposit = get_value(df, year, 'Deposit income', keycodename='Deposit income')
                    affiliates = get_value(df, year, 'Gain/loss from affiliates divestment', keycodename='Gain/(loss) in affliates divestments')
                    associates = get_value(df, year, 'Income from associate companies', keycodename='Income from associate companies')
                    net_other = get_value(df, year, 'Net other income', keycodename='Net other income')
                    values.append(fx + deposit + affiliates + associates + net_other)
                data[item_name] = values
            elif item_name == 'Interest expense':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Interest expense', keycodename='Interest Expense'))
                data[item_name] = values
            elif item_name == 'Borrowing balance':
                values = []
                for year in years:
                    if year == 2025:
                        # For 2025, calculate as sum of short-term and long-term borrowing from FORECAST.csv
                        short_term_borrowing = get_value(df_broker_forecast, year, None, keycodename='Short-term borrowing')
                        long_term_borrowing = get_value(df_broker_forecast, year, None, keycodename='Long-term borrowing')
                        total_borrowing = short_term_borrowing + long_term_borrowing
                        values.append(total_borrowing)
                    else:
                        # For historical years, use the existing logic
                        values.append(get_value(df, year, 'Borrowing balance', keycodename='Borrowing balance'))
                data[item_name] = values
            elif item_name == 'Tax expense':
                values = []
                for year in years:
                    values.append(get_value(df, year, 'Tax expense', keycodename='Tax expense'))
                data[item_name] = values
            elif item_name == 'CIR':
                values = []
                for year in years:
                    if year == 2025:
                        values.append(cir_adj/100)
                    elif year == 2026:
                        values.append(cir_2026 if 'cir_2026' in locals() else cir_adj/100)
                    else:
                        # For historical years: CIR = SG&A / (Total operating income - Investment income)
                        sga = get_value(df, year, 'SG&A', keycodename='SG&A')
                        total_operating_income = data.get('Total Operating Income', [0]*len(years))[years.index(year)] if 'Total Operating Income' in data else 0
                        investment_income = data.get('Net Investment Income', [0]*len(years))[years.index(year)] if 'Net Investment Income' in data else 0
                        denominator = total_operating_income - investment_income
                        if denominator != 0:
                            cir = abs(sga) / denominator  # Use abs(sga) since SG&A is typically negative
                        else:
                            cir = 0
                        values.append(cir)
                data[item_name] = values
            elif item_name == 'PBT':
                values = []
                for i, year in enumerate(years):
                    toi = data.get('Total Operating Income', [0]*len(years))[i] if 'Total Operating Income' in data else 0
                    # Use already-calculated SG&A value from data dict for consistency (especially for 2025 dynamic calculation)
                    sga = data.get('SG&A', [0]*len(years))[i] if 'SG&A' in data else get_value(df, year, 'SG&A', keycodename='SG&A')
                    # Use already-calculated value for 'Other incomes' from data dict
                    others = data.get('Other incomes', [0]*len(years))[i] if 'Other incomes' in data else 0
                    interest = get_value(df, year, 'Interest expense', keycodename='Interest Expense')
                    values.append(toi + sga + others + interest)
                data[item_name] = values
            elif item_name == 'NPAT':
                values = []
                for year in years:
                    pbt = data.get('PBT', [0]*len(years))[years.index(year)] if 'PBT' in data else 0
                    tax = get_value(df, year, 'Tax expense', keycodename='Tax expense')
                    values.append(pbt + tax)
                data[item_name] = values
            elif item_name == 'Capital Income':
                values = []
                for i, year in enumerate(years):
                    if year == 2025:
                        # For forecast year, calculate from components using dynamically calculated values
                        investment = get_value(df, year, None, keycodename='Net Investment Income')
                        margin = data.get('Margin Lending Income', [0]*len(years))[i]  # Use the dynamically calculated margin income
                        capital = investment + margin
                    else:
                        # For historical years, use the pre-calculated value
                        capital = get_value(df, year, 'Capital Income', keycodename=None)
                    values.append(capital)
                data[item_name] = values
            elif item_name == 'Total Operating Income':
                values = []
                for i, year in enumerate(years):
                    # Use dynamically calculated values from data dictionary
                    capital = data.get('Capital Income', [0]*len(years))[i]
                    fee = data.get('Net Fees Income', [0]*len(years))[i]
                    toi = capital + fee
                    values.append(toi)
                data[item_name] = values
            elif item_name == 'SG&A':
                values = []
                for i, year in enumerate(years):
                    if year == 2025:
                        # Calculate SG&A dynamically using CIR formula: SG&A = CIR × (Total Operating Income - Net Investment Income)
                        total_op_income_2025 = data.get('Total Operating Income', [0]*len(years))[i]
                        net_inv_income_2025 = data.get('Net Investment Income', [0]*len(years))[i]                       
                        sga_calculated = (cir_adj_to_use / 100.0) * (total_op_income_2025 - net_inv_income_2025)
                        
                        values.append(-abs(sga_calculated))  # SG&A is typically negative
                    elif year == 2026:
                        # Calculate SG&A for 2026 using 2026 CIR
                        total_op_income_2026 = data.get('Total Operating Income', [0]*len(years))[i]
                        net_inv_income_2026 = data.get('Net Investment Income', [0]*len(years))[i]
                        cir_2026_val = cir_2026 if 'cir_2026' in locals() else cir_adj_to_use / 100.0
                        sga_calculated_2026 = cir_2026_val * (total_op_income_2026 - net_inv_income_2026)
                        
                        values.append(-abs(sga_calculated_2026))  # SG&A is typically negative
                    else:
                        values.append(get_value(df, year, 'SG&A', keycodename='SG&A'))
                data[item_name] = values
            # Add similar blocks for other items as needed
            else:
                # Handle lambda functions for other items
                if filter_func is not None:
                    values = []
                    for year in years:
                        year_data = df[df['DATE'] == year]
                        value = filter_func(year_data)
                        values.append(value)
                    data[item_name] = values
        result = pd.DataFrame(data, index=years).T
        result.columns = [str(y) for y in years]
        result.index.name = "Item"

        # Format all values as billions with thousand separators, except market share (show as percent)
        for item in result.index:
            if item == 'Market Turnover':
                for col in result.columns:
                    val = result.at[item, col]
                    if isinstance(val, str):
                        try:
                            val_num = float(val)
                        except ValueError:
                            val_num = None
                    else:
                        val_num = val
                    if pd.notnull(val_num) and pd.api.types.is_number(val_num):
                        result.at[item, col] = f"{val_num:,.0f}"
                    else:
                        result.at[item, col] = "-"
            elif item == 'Brokerage Market Share':
                for col in result.columns:
                    val = result.at[item, col]
                    result.at[item, col] = f"{val:.2%}" if pd.notnull(val) and pd.api.types.is_number(val) else "-"
            elif item == 'Net Brokerage Fee':
                for col in result.columns:
                    val = result.at[item, col]
                    if pd.notnull(val) and pd.api.types.is_number(val):
                        result.at[item, col] = f"{val:.3%}"
                    else:
                        result.at[item, col] = "-"
            elif item == 'Company Turnover' or item == 'Market Turnover':
                for i, col in enumerate(result.columns):
                    val = result.at[item, col]
                    # Try to convert string numbers to float
                    if isinstance(val, str):
                        try:
                            val_num = float(val)
                        except ValueError:
                            val_num = None
                    else:
                        val_num = val
                    # Only divide by 1e9 for 2025 (last column)
                    if pd.notnull(val_num) and pd.api.types.is_number(val_num):
                        if col == '2025':
                            result.at[item, col] = f"{val_num:,.0f}"
                        else:
                            result.at[item, col] = f"{val_num:,.0f}"
                    else:
                        result.at[item, col] = "-"
            elif item == 'Lending rate':
                for col in result.columns:
                    val = result.at[item, col]
                    if pd.notnull(val) and pd.api.types.is_number(val):
                        result.at[item, col] = f"{val:.2%}"
                    else:
                        result.at[item, col] = "-"
            elif item == 'CIR':
                for col in result.columns:
                    val = result.at[item, col]
                    if pd.notnull(val) and pd.api.types.is_number(val):
                        result.at[item, col] = f"{val:.2%}"
                    else:
                        result.at[item, col] = "-"
            else:
                for col in result.columns:
                    val = result.at[item, col]
                    if isinstance(val, str):
                        try:
                            val_num = float(val)
                        except ValueError:
                            val_num = None
                    else:
                        val_num = val
                    if pd.notnull(val_num) and pd.api.types.is_number(val_num):
                        result.at[item, col] = f"{val_num/1e9:,.0f}"
                    else:
                        result.at[item, col] = "-"
        return result
    # Build historical financial statement from custom calculations
    historical_years = sorted(filtered_is['Year'].unique())
    year_start = 2020
    year_end = current_year
    years_to_display = list(range(year_start, year_end + 1))

    # Helper to align values by year
    def align_by_year(df, col, years):
        # Ensure 'Year' column and years are both integers
        df['Year'] = df['Year'].astype(int)
        df = df.set_index('Year')
        return [df.at[y, col] if y in df.index else 0 for y in years]

    historical_data = {
        'DATE': years_to_display,
        'Net Brokerage Income': align_by_year(net_brokerage, 'Net Brokerage Income', years_to_display),
        'Net IB Income': align_by_year(ib_inc, 'Net IB Income', years_to_display),
        'Net other operating income': align_by_year(other_op_inc, 'Net other operating income', years_to_display),
        'Net Investment Income': align_by_year(inv_inc, 'Net Investment Income', years_to_display),
        'Capital Income': align_by_year(capital_inc, 'Capital Income', years_to_display),
        'Total Operating Income': align_by_year(total_operating_income, 'Total Operating Income', years_to_display),
        'SG&A': align_by_year(sga, 'SG&A', years_to_display),
        'PBT': align_by_year(pbt, 'PBT', years_to_display),
        'NPAT': align_by_year(NPAT, 'NPAT', years_to_display),
        'Margin Lending Income': align_by_year(margin_inc, 'Margin Lending Income', years_to_display),
        'Net other income': align_by_year(other_inc, 'Net other income', years_to_display),
        'Deposit income': align_by_year(deposit_inc, 'Deposit income', years_to_display),
        'Gain/loss from affiliates divestment': align_by_year(filtered_is.groupby('Year')[['IS.46']].sum().reset_index().rename(columns={'IS.46': 'Gain/loss from affiliates divestment'}), 'Gain/loss from affiliates divestment', years_to_display),
        'Income from associate companies': align_by_year(filtered_is.groupby('Year')[['IS.47','IS.55']].sum().sum(axis=1).reset_index(name='Income from associate companies'), 'Income from associate companies', years_to_display),
        'Interest expense': align_by_year(interest_exp, 'Interest expense', years_to_display),
        'Borrowing balance': align_by_year(borrowing_balance, 'Borrowing Balance', years_to_display),
        'Tax expense': align_by_year(filtered_is.groupby('Year')[['IS.68']].sum().reset_index().rename(columns={'IS.68': 'Tax expense'}), 'Tax expense', years_to_display),
        'Minority expense': align_by_year(filtered_is.groupby('Year')[['IS.70']].sum().reset_index().rename(columns={'IS.70': 'Minority expense'}), 'Minority expense', years_to_display),
        'Margin Balance': align_by_year(margin_balance, 'Margin Balance', years_to_display),
        'FX gain/loss': align_by_year(fx_gain_loss_by_year, 'FX gain/loss', years_to_display),
        # Add more items as needed from your calculated DataFrames
    }
    df_broker_hist_custom = pd.DataFrame(historical_data)
    # For forecast year, keep using FORECAST.csv/sidebar logic
    if st.session_state.show_2026_forecast:
        df_broker_all = pd.concat([df_broker_hist_custom, df_broker_forecast, df_broker_forecast_2026], ignore_index=True)
    else:
        df_broker_all = pd.concat([df_broker_hist_custom, df_broker_forecast], ignore_index=True)

    # --- Key Performance Metrics ---
    # Display broker name at top
    st.title(f"{selected_broker}")
    
    # Create placeholder for metrics that will be updated after all controls
    metrics_placeholder = st.empty()
    
    st.divider()
    
    # --- Market Turnover Summary ---
    today = pd.Timestamp.today()
    current_year = today.year
    current_quarter = today.quarter
    # Filter for current year
    df_index_current_year = df_index[df_index['Year'] == current_year]
    # Previous quarters (exclude current quarter)
    prev_quarters = [f"{current_year}Q{q}" for q in range(1, current_quarter)]
    avg_daily_prev_quarters = {}
    for q in prev_quarters:
        df_q = df_index_current_year[df_index_current_year['Quarter'] == q]
        if not df_q.empty:
            avg_daily_prev_quarters[q] = df_q['TOTALVALUE'].mean() / 1e9
    # Current quarter qtd (from beginning of quarter to latest available date)
    current_q_str = f"{current_year}Q{current_quarter}"
    df_current_q = df_index_current_year[df_index_current_year['Quarter'] == current_q_str]
    if not df_current_q.empty:
        # Get all data from beginning of current quarter to latest available date
        avg_daily_qtd = df_current_q['TOTALVALUE'].mean() / 1e9
    else:
        avg_daily_qtd = None
    # YTD
    avg_daily_ytd = df_index_current_year['TOTALVALUE'].mean() / 1e9 if not df_index_current_year.empty else None

    # Build pivoted market turnover summary table
    turnover_dict = {}
    for q, val in avg_daily_prev_quarters.items():
        # Format with thousand separators
        turnover_dict[q] = f"{val:,.0f}"
    if avg_daily_qtd is not None:
        turnover_dict[f"{current_q_str} QTD"] = f"{avg_daily_qtd:,.0f}"
    if avg_daily_ytd is not None:
        turnover_dict[f"YTD {current_year}"] = f"{avg_daily_ytd:,.0f}"
    # Ensure 2025Q1 is present and aligned
    if f"{current_year}Q1" not in turnover_dict:
        df_q1 = df_index_current_year[df_index_current_year['Quarter'] == f"{current_year}Q1"]
        if not df_q1.empty:
            val_q1 = df_q1['TOTALVALUE'].mean() / 1e9
            turnover_dict[f"{current_year}Q1"] = f"{val_q1:,.0f}"
    # Sort keys for display
    sorted_periods = sorted(turnover_dict.keys(), key=lambda x: (x.split()[0], x))
    turnover_df = pd.DataFrame([turnover_dict])
    turnover_df = turnover_df.rename_axis('Metric').T
    turnover_df.columns = ['Avg Daily Market Turnover (VNDbn)']
    turnover_df.index.name = 'Period'
    turnover_df = turnover_df.loc[sorted_periods]
    st.subheader("Market Turnover Summary")
    st.dataframe(turnover_df.T)

    # --- Financial Statement ---
    statement_pivot = build_financial_statement_pivot(df_broker_all, items)
    
    # Hide specified rows from display
    rows_to_hide = [
        'Net other operating income',
        'Net other income',
        'Gain/loss from affiliates divestment',
        'Tax expense',
        'FX gain/loss',
        'Deposit income',
        'Income from associate companies',
        'Other incomes',
        'Net Fees Income',
        'Capital Income',
    ]
    
    # Define financial statement chunks (including hidden rows for completeness)
    net_brokerage_chunk = [
        'Net Brokerage Income',
        'Market Turnover', 
        'Company Turnover',
        'Brokerage Market Share',
        'Net Brokerage Fee'
    ]
    
    margin_lending_chunk = [
        'Margin Lending Income',
        'Margin Balance',
        'Lending rate'
    ]
    
    expense_chunk = [
        'SG&A',
        'CIR',
        'Interest expense',
        'Borrowing balance',
        'Other Expenses'
    ]
    
    other_income_chunk = [
        'Net IB Income',
        'Net Investment Income',
        'Net other operating income',  # Will be hidden
        'FX gain/loss',                # Will be hidden
        'Deposit income',              # Will be hidden
        'Gain/loss from affiliates divestment',  # Will be hidden
        'Income from associate companies',       # Will be hidden
        'Net other income'             # Will be hidden
    ]
    
    summary_chunk = [
        'Total Operating Income',
        'PBT',
        'NPAT'
    ]
    
    # Function to filter out hidden rows from chunks
    def filter_chunk(chunk_items, rows_to_hide):
        return [item for item in chunk_items if item not in rows_to_hide]
    
    # Display each chunk separately with hidden rows filtered out
    st.subheader("Brokerage")
    
    # Brokerage adjustment controls
    col1, col2, col3 = st.columns(3)
    with col1:
        market_share_adj = st.number_input("Company Market Share (%)", value=market_share_2024_default, format="%.1f", key="market_share_brokerage")
    with col2:
        market_turnover_daily_adj = st.number_input("Market Turnover per Day (VNDbn)", min_value=0.0, max_value=100000.0, value=market_turnover_daily_2024_default, step=100.0, format="%.0f", key="market_turnover_brokerage")
    with col3:
        net_fee_adj = st.number_input("Net Brokerage Fee (%)", value=net_fee_2024_default * 100, format="%.2f", key="net_fee_brokerage")
    
    # 2026 Brokerage controls (conditional)
    if st.session_state.show_2026_forecast:
        st.write("**2026 Adjustments**")
        col1_2026, col2_2026, col3_2026 = st.columns(3)
        with col1_2026:
            market_share_2026_adj = st.number_input("2026 Market Share (%)", value=market_share_2024_default, format="%.1f", key="market_share_2026_brokerage")
        with col2_2026:
            market_turnover_2026_daily_adj = st.number_input("2026 Market Turnover/Day (VNDbn)", min_value=0.0, max_value=100000.0, value=market_turnover_daily_2024_default, step=100.0, format="%.0f", key="market_turnover_2026_brokerage")
        with col3_2026:
            net_fee_2026_adj = st.number_input("2026 Net Brokerage Fee (%)", value=net_fee_2024_default * 100, format="%.2f", key="net_fee_2026_brokerage")
    else:
        # Set default values when 2026 is not shown
        market_share_2026_adj = market_share_2024_default
        market_turnover_2026_daily_adj = market_turnover_daily_2024_default
        net_fee_2026_adj = net_fee_2024_default * 100
    
    # Update calculated values based on user inputs
    market_share_2025 = market_share_adj / 100.0
    market_turnover_2025 = market_turnover_daily_adj * trading_days_2025
    company_turnover_2025 = market_share_2025 * market_turnover_2025 * 2
    net_fee_2025 = net_fee_adj / 100.0
    
    # Update 2026 calculated values
    market_share_2026 = market_share_2026_adj / 100.0
    market_turnover_2026 = market_turnover_2026_daily_adj * trading_days_2025  # Assume same trading days
    company_turnover_2026 = market_share_2026 * market_turnover_2026 * 2
    net_fee_2026 = net_fee_2026_adj / 100.0
    
    # Rebuild the statement pivot with updated values
    statement_pivot = build_financial_statement_pivot(df_broker_all, items)
    
    net_brokerage_filtered = filter_chunk(net_brokerage_chunk, rows_to_hide)
    net_brokerage_data = statement_pivot.loc[statement_pivot.index.intersection(net_brokerage_filtered)]
    if not net_brokerage_data.empty:
        st.dataframe(net_brokerage_data, height=210)
    
    st.subheader("Margin Lending")
    
    # Margin lending adjustment control
    margin_adj = st.number_input("Margin Balance (VNDbn)", min_value=0.0, max_value=100000.0, value=margin_2024_default, step=100.0, format="%.0f", key="margin_lending")
    
    # 2026 Margin Lending controls (conditional)
    if st.session_state.show_2026_forecast:
        st.write("**2026 Adjustments**")
        margin_2026_adj = st.number_input("2026 Margin Balance (VNDbn)", min_value=0.0, max_value=100000.0, value=margin_2024_default, step=100.0, format="%.0f", key="margin_2026_lending")
    else:
        # Set default value when 2026 is not shown
        margin_2026_adj = margin_2024_default
    
    # Update 2026 margin value
    margin_2026 = margin_2026_adj
    
    # Rebuild the statement pivot again with updated margin balance
    statement_pivot = build_financial_statement_pivot(df_broker_all, items)
    
    # Now calculate and display the key metrics with all user adjustments
    # Get current adjusted PBT for 2025
    current_pbt_2025 = statement_pivot.loc['PBT', '2025'] if 'PBT' in statement_pivot.index else 0
    try:
        current_pbt_value = float(str(current_pbt_2025).replace(",", "")) * 1e9  # Convert to actual value
        current_pbt_display = current_pbt_value / 1e9  # Convert to billions for display
    except:
        current_pbt_value = 0
        current_pbt_display = 0
    
    # Get 2024 PBT for YoY growth calculation
    pbt_2024 = statement_pivot.loc['PBT', '2024'] if 'PBT' in statement_pivot.index and '2024' in statement_pivot.columns else 0
    try:
        pbt_2024_value = float(str(pbt_2024).replace(",", "")) * 1e9 if pbt_2024 != 0 else 0
        yoy_growth = ((current_pbt_value - pbt_2024_value) / pbt_2024_value * 100) if pbt_2024_value != 0 else 0
    except:
        yoy_growth = 0
    
    # Get baseline PBT from FORECAST.csv
    baseline_pbt_header = get_value(df_broker_forecast, 2025, 'PBT', keycodename='PBT')
    baseline_change = ((current_pbt_value - baseline_pbt_header) / baseline_pbt_header * 100) if baseline_pbt_header != 0 else 0
    
    # Update the metrics placeholder with actual calculated values
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="2025 Baseline PBT",
                value=f"{baseline_pbt_header/1e9:,.0f}B",
            )
        with col2:
            st.metric(
                label="2025 Adjusted PBT",
                value=f"{current_pbt_display:,.0f}B",
                delta=f"{baseline_change:+.1f}% vs baseline"
            )
        with col3:
            st.metric(
                label="YoY Growth",
                value=f"{yoy_growth:+.1f}%",
                delta="vs 2024"
            )
    
    margin_lending_filtered = filter_chunk(margin_lending_chunk, rows_to_hide)
    margin_lending_data = statement_pivot.loc[statement_pivot.index.intersection(margin_lending_filtered)]
    if not margin_lending_data.empty:
        st.dataframe(margin_lending_data, height=140)
    
    st.subheader("Expenses")
    cir_adj = st.number_input("CIR (%)", value=cir_2025_default_percent, format="%.1f", key="cir_expenses")
    
    # 2026 CIR controls (conditional)
    if st.session_state.show_2026_forecast:
        st.write("**2026 Adjustments**")
        cir_2026_adj = st.number_input("2026 CIR (%)", value=cir_2025_default_percent, format="%.1f", key="cir_2026_expenses")
    else:
        # Set default value when 2026 is not shown
        cir_2026_adj = cir_2025_default_percent
    
    # Update 2026 CIR value
    cir_2026 = cir_2026_adj / 100.0
    
    # Rebuild statement_pivot with the CIR adjustment
    statement_pivot = build_financial_statement_pivot(df_broker_all, items, cir_adj_param=cir_adj)
    
    expense_filtered = filter_chunk(expense_chunk, rows_to_hide)
    expense_data = statement_pivot.loc[statement_pivot.index.intersection(expense_filtered)]
    if not expense_data.empty:
        st.dataframe(expense_data, height=200)

    st.subheader("Other Income Items")
    other_income_filtered = filter_chunk(other_income_chunk, rows_to_hide)
    others_data = statement_pivot.loc[statement_pivot.index.intersection(other_income_filtered)]
    if not others_data.empty:
        st.dataframe(others_data, height=107)
    
    st.subheader("Summary")
    summary_filtered = filter_chunk(summary_chunk, rows_to_hide)
    summary_data = statement_pivot.loc[statement_pivot.index.intersection(summary_filtered)]
    if not summary_data.empty:
        st.dataframe(summary_data, height=143)
    
else:
    st.warning("No data available for the selected period.")