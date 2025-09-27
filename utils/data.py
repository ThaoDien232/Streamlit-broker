import streamlit as st
import pandas as pd
from datetime import datetime
from utils.db import run_query

@st.cache_data(ttl=600)  # cache for 10 minutes
def get_broker_history(broker: str, start_date: str):
    sql = """
        SELECT trade_date, metric, value
        FROM broker_history
        WHERE broker = :broker AND trade_date >= :start_date
        ORDER BY trade_date;
    """
    return run_query(sql, {"broker": broker, "start_date": start_date})


def calculate_income_statement_items(filtered_is, filtered_bs):
    """
    Calculate income statement items from financial data.
    
    Args:
        filtered_is: DataFrame with filtered income statement data
        filtered_bs: DataFrame with filtered balance sheet data
        
    Returns:
        Dict containing calculated financial metrics by year
    """
    # FX gain/loss
    fx_gain_loss_by_year = filtered_is.groupby('Year')[['IS.44','IS.50']].sum().sum(axis=1).reset_index(name='FX gain/loss')
    
    # Affiliates
    affiliates = filtered_is.groupby('Year')[['IS.46']].sum().reset_index().rename(columns={'IS.46': 'Gain/loss from affiliates divestment'})
    
    # Associates
    associates = filtered_is.groupby('Year')[['IS.47','IS.55']].sum().sum(axis=1).reset_index(name='Income from associate companies')
    
    # Deposit income
    deposit_inc = filtered_is.groupby('Year')[['IS.45']].sum().reset_index().rename(columns={'IS.45': 'Deposit income'})
    
    # Interest expense
    interest_exp = filtered_is.groupby('Year')[['IS.51']].sum().reset_index().rename(columns={'IS.51': 'Interest expense'})
    
    # Investment banking income
    ib_inc = filtered_is.groupby('Year')[['IS.12','IS.13','IS.15','IS.11','IS.16','IS.17','IS.18','IS.34','IS.35','IS.36','IS.38']].sum().sum(axis=1).reset_index(name='Net IB Income')
    
    # Net brokerage income
    net_brokerage = filtered_is.groupby('Year')[['IS.10','IS.33']].sum().sum(axis=1).reset_index(name='Net Brokerage Income')
    
    # Trading income
    trading_inc = filtered_is.groupby('Year')[['IS.3','IS.4','IS.5','IS.8','IS.9','IS.27','IS.24','IS.25','IS.26','IS.28','IS.29','IS.31','IS.32']].sum().sum(axis=1).reset_index(name='Net trading income')
    
    # Interest income
    interest_inc = filtered_is.groupby('Year')[['IS.6']].sum().reset_index().rename(columns={'IS.6': 'Interest income'})
    
    # Investment income
    inv_inc = trading_inc.merge(interest_inc, on='Year', how='outer')
    inv_inc = inv_inc.fillna(0)
    inv_inc['Net Investment Income'] = inv_inc['Net trading income'] + inv_inc['Interest income']
    
    # Margin lending income
    margin_inc = filtered_is.groupby('Year')[['IS.7','IS.30']].sum().sum(axis=1).reset_index(name='Margin Lending Income')
    
    # Other income
    other_inc = filtered_is.groupby('Year')[['IS.52','IS.54','IS.63']].sum().sum(axis=1).reset_index(name='Net other income')
    
    # Other operating income
    other_op_inc = filtered_is.groupby('Year')[['IS.14','IS.19','IS.20','IS.37','IS.39','IS.40']].sum().sum(axis=1).reset_index(name='Net other operating income')
    
    # Fee income calculation
    fee_inc = net_brokerage[['Year', 'Net Brokerage Income']].merge(
        ib_inc[['Year', 'Net IB Income']], on='Year', how='outer').merge(
        other_op_inc[['Year', 'Net other operating income']], on='Year', how='outer')
    fee_inc = fee_inc.fillna(0)
    fee_inc['Fee Income'] = fee_inc['Net Brokerage Income'] + fee_inc['Net IB Income'] + fee_inc['Net other operating income']
    
    # Capital income calculation
    capital_inc = trading_inc[['Year', 'Net trading income']].merge(
        interest_inc[['Year', 'Interest income']], on='Year', how='outer').merge(
        margin_inc[['Year', 'Margin Lending Income']], on='Year', how='outer')
    capital_inc = capital_inc.fillna(0)
    capital_inc['Capital Income'] = capital_inc['Net trading income'] + capital_inc['Interest income'] + capital_inc['Margin Lending Income']
    
    # Total operating income calculation
    total_operating_income = capital_inc[['Year', 'Capital Income']].merge(
        fee_inc[['Year', 'Fee Income']], on='Year', how='outer')
    total_operating_income = total_operating_income.fillna(0)
    total_operating_income['Total Operating Income'] = total_operating_income['Capital Income'] + total_operating_income['Fee Income']
    
    # Borrowing balance
    borrowing_balance = filtered_bs.groupby('Year')[['BS.95','BS.100','BS.122','BS.127']].sum().sum(axis=1).reset_index(name='Borrowing Balance')
    
    # SG&A
    sga = filtered_is.groupby('Year')[['IS.57','IS.58']].sum().sum(axis=1).reset_index(name='SG&A')
    
    # PBT
    pbt = filtered_is.groupby('Year')[['IS.65']].sum().reset_index().rename(columns={'IS.65': 'PBT'})
    
    # NPAT
    NPAT = filtered_is.groupby('Year')[['IS.71']].sum().reset_index().rename(columns={'IS.71': 'NPAT'})
    
    # Margin balance
    margin_balance = filtered_bs.sort_values('ENDDATE').groupby('Year').tail(1)[['Year', 'BS.8']].rename(columns={'BS.8': 'Margin Balance'})
    
    # Total equity from BS_security using OWNER'S EQUITY (BS.142)
    total_equity = filtered_bs.groupby('Year')[['BS.142']].sum().reset_index().rename(columns={'BS.142': 'Total Equity'})
    
    # Investment asset balance = short-term investments + available-for-sale assets + available-for-sale securities + held-to-maturity assets
    short_term_investments = filtered_bs.groupby('Year')[['BS.6']].sum().reset_index().rename(columns={'BS.6': 'Short-term Investments'})
    available_for_sale_assets = filtered_bs.groupby('Year')[['BS.9']].sum().reset_index().rename(columns={'BS.9': 'Available-for-Sale Assets'})
    available_for_sale_securities = filtered_bs.groupby('Year')[['BS.58']].sum().reset_index().rename(columns={'BS.58': 'Available-for-Sale Securities'})
    held_to_maturity_short = filtered_bs.groupby('Year')[['BS.7']].sum().reset_index().rename(columns={'BS.7': 'Held-to-Maturity Short'})
    held_to_maturity_long = filtered_bs.groupby('Year')[['BS.59']].sum().reset_index().rename(columns={'BS.59': 'Held-to-Maturity Long'})
    
    # Combine all investment asset components
    investment_asset_balance = short_term_investments.merge(available_for_sale_assets, on='Year', how='outer').merge(
        available_for_sale_securities, on='Year', how='outer').merge(
        held_to_maturity_short, on='Year', how='outer').merge(
        held_to_maturity_long, on='Year', how='outer')
    investment_asset_balance = investment_asset_balance.fillna(0)
    investment_asset_balance['Investment Asset Balance'] = (
        investment_asset_balance['Short-term Investments'] + 
        investment_asset_balance['Available-for-Sale Assets'] + 
        investment_asset_balance['Available-for-Sale Securities'] + 
        investment_asset_balance['Held-to-Maturity Short'] + 
        investment_asset_balance['Held-to-Maturity Long']
    )
    
    # Investment return % = net investment income / average (this year investment asset + previous year investment asset)
    inv_return = inv_inc.merge(investment_asset_balance[['Year', 'Investment Asset Balance']], on='Year', how='outer')
    inv_return = inv_return.fillna(0)
    inv_return['Investment Return %'] = 0.0  # Initialize
    
    # Calculate Investment Return % using average of current and previous year assets
    for idx, row in inv_return.iterrows():
        current_year_assets = row['Investment Asset Balance']
        net_investment_income = row['Net Investment Income']
        
        # Get previous year assets
        if idx > 0:
            previous_year_assets = inv_return.iloc[idx-1]['Investment Asset Balance']
        else:
            previous_year_assets = current_year_assets  # For first year, use current year
        
        # Calculate average investment assets
        average_investment_assets = (current_year_assets + previous_year_assets) / 2 if previous_year_assets > 0 else current_year_assets
        
        # Calculate investment return percentage (as decimal for percentage formatting)
        if average_investment_assets > 0:
            inv_return.at[idx, 'Investment Return %'] = net_investment_income / average_investment_assets
    
    investment_return = inv_return[['Year', 'Investment Return %']]
    
    # Net Margin Income with 10% provision rule
    # Components: Income from loans and receivables (IS.6) + Provision for losses (IS.30)
    loans_income = filtered_is.groupby('Year')[['IS.6']].sum().reset_index().rename(columns={'IS.6': 'Income from loans and receivables'})
    provision_losses = filtered_is.groupby('Year')[['IS.30']].sum().reset_index().rename(columns={'IS.30': 'Provision for losses'})
    
    # Merge the two components
    net_margin_base = loans_income.merge(provision_losses, on='Year', how='outer')
    net_margin_base = net_margin_base.fillna(0)
    
    # Apply 10% rule: If provision/loans > 10%, use loans income only
    net_margin_base['Net Margin Income'] = net_margin_base.apply(
        lambda row: row['Income from loans and receivables'] 
        if (row['Income from loans and receivables'] != 0 and 
            abs(row['Provision for losses'] / row['Income from loans and receivables']) > 0.1)
        else row['Income from loans and receivables'] + row['Provision for losses'], 
        axis=1
    )
    
    net_margin_income = net_margin_base[['Year', 'Net Margin Income']]
    
    # Total Assets for ROA calculation
    total_assets = filtered_bs.groupby('Year')[['BS.92']].sum().reset_index().rename(columns={'BS.92': 'Total Assets'})
    
    # ROE = NPAT / average (this year + last year Total equity)
    roe_data = NPAT.merge(total_equity, on='Year', how='outer')
    roe_data = roe_data.fillna(0)
    roe_data['ROE'] = 0.0  # Initialize
    
    for idx, row in roe_data.iterrows():
        current_equity = row['Total Equity']
        npat = row['NPAT']
        
        # Get previous year equity
        if idx > 0:
            previous_equity = roe_data.iloc[idx-1]['Total Equity']
        else:
            previous_equity = current_equity  # For first year, use current year
        
        # Calculate average equity
        average_equity = (current_equity + previous_equity) / 2 if previous_equity > 0 else current_equity
        
        # Calculate ROE
        if average_equity > 0:
            roe_data.at[idx, 'ROE'] = npat / average_equity
    
    roe = roe_data[['Year', 'ROE']]
    
    # ROA = NPAT / average (this year + last year Total assets)
    roa_data = NPAT.merge(total_assets, on='Year', how='outer')
    roa_data = roa_data.fillna(0)
    roa_data['ROA'] = 0.0  # Initialize
    
    for idx, row in roa_data.iterrows():
        current_assets = row['Total Assets']
        npat = row['NPAT']
        
        # Get previous year assets
        if idx > 0:
            previous_assets = roa_data.iloc[idx-1]['Total Assets']
        else:
            previous_assets = current_assets  # For first year, use current year
        
        # Calculate average assets
        average_assets = (current_assets + previous_assets) / 2 if previous_assets > 0 else current_assets
        
        # Calculate ROA
        if average_assets > 0:
            roa_data.at[idx, 'ROA'] = npat / average_assets
    
    roa = roa_data[['Year', 'ROA']]
    
    return {
        'fx_gain_loss_by_year': fx_gain_loss_by_year,
        'affiliates': affiliates,
        'associates': associates,
        'deposit_inc': deposit_inc,
        'interest_exp': interest_exp,
        'ib_inc': ib_inc,
        'net_brokerage': net_brokerage,
        'trading_inc': trading_inc,
        'interest_inc': interest_inc,
        'inv_inc': inv_inc,
        'margin_inc': margin_inc,
        'other_inc': other_inc,
        'other_op_inc': other_op_inc,
        'fee_inc': fee_inc,
        'capital_inc': capital_inc,
        'total_operating_income': total_operating_income,
        'borrowing_balance': borrowing_balance,
        'sga': sga,
        'pbt': pbt,
        'NPAT': NPAT,
        'margin_balance': margin_balance,
        'total_equity': total_equity,
        'investment_asset_balance': investment_asset_balance,
        'investment_return': investment_return,
        'net_margin_income': net_margin_income,
        'total_assets': total_assets,
        'roe': roe,
        'roa': roa
    }


def calculate_market_turnover_and_trading_days(df_index, current_year=None):
    """
    Calculate average daily market turnover and trading days for each year from INDEX file.
    
    Args:
        df_index: DataFrame containing index data with TRADINGDATE and TOTALVALUE columns
        current_year: Current year (defaults to current datetime year)
        
    Returns:
        Tuple containing (market_turnover_by_year DataFrame, trading_days_2025)
    """
    if current_year is None:
        current_year = datetime.now().year
    
    # Extract year from trading date
    df_index = df_index.copy()
    df_index['Year'] = pd.to_datetime(df_index['TRADINGDATE']).dt.year

    # Calculate market turnover by year
    market_turnover_by_year = df_index.groupby('Year').agg(
        trading_days=('TRADINGDATE', 'nunique'),
        total_turnover=('TOTALVALUE', 'sum')
    ).reset_index()
    
    # Calculate average daily turnover for each year
    market_turnover_by_year['avg_daily_turnover'] = market_turnover_by_year['total_turnover'] / market_turnover_by_year['trading_days']

    # Get trading days for next year - use current year as reference since next year data won't exist yet
    trading_days_current = market_turnover_by_year.loc[market_turnover_by_year['Year'] == current_year, 'trading_days']
    if not trading_days_current.empty:
        trading_days_next_year = int(trading_days_current.iloc[0])  # Use current year trading days as baseline
    else:
        trading_days_next_year = 252  # fallback typical value
    
    return market_turnover_by_year, trading_days_next_year
