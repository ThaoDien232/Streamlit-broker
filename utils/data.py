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
        'margin_balance': margin_balance
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
