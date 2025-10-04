"""
Data loading utilities for brokerage sector valuation analysis.
Connects to SQL Server database to fetch market data and calculate valuation metrics.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Optional
import pandas as pd
import streamlit as st
from utils.db import run_query, test_connection, get_available_tables

def _load_dataframe_from_db(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Load dataframe using existing database connection."""
    try:
        return run_query(query, params)
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        print(f"Database query failed: {e}")
        print(f"Query: {query}")
        return pd.DataFrame()

def get_broker_tickers() -> list:
    """Get list of all brokerage tickers from the database or use default list."""

    # Default list of Vietnamese broker tickers
    default_tickers = [
        'SSI', 'VCI', 'HCM', 'VIX', 'VND', 'MBS', 'SHS', 'BSI',
        'TCBS', 'FPTS', 'AGR', 'CTS', 'VDS', 'APS', 'ORS', 'PSI',
        'BVS', 'IVS', 'EVS', 'VFS', 'VIG', 'TVS', 'CVT', 'AAS',
        'WSS', 'TCI', 'DSC', 'VRS', 'VTS', 'VSC', 'SHB', 'TGG'
    ]

    try:
        # Try to get actual tickers from database if possible
        query = """
        SELECT DISTINCT TICKER
        FROM dbo.Market_Data
        WHERE TICKER IN ({})
        AND TRADE_DATE >= DATEADD(year, -1, GETDATE())
        ORDER BY TICKER
        """.format(','.join([f"'{t}'" for t in default_tickers]))

        result = run_query(query)
        if not result.empty:
            return sorted(result['TICKER'].tolist())

    except Exception as e:
        print(f"Could not fetch tickers from database: {e}")

    return sorted(default_tickers)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def check_database_schema() -> dict:
    """Check what columns are available in the Market_Data table."""
    schema_info = {
        'table_exists': False,
        'available_columns': [],
        'sample_data': pd.DataFrame()
    }

    try:
        # Check if Market_Data table exists
        tables_df = get_available_tables()
        market_tables = tables_df[
            tables_df['TABLE_NAME'].str.contains('Market', case=False, na=False)
        ]['TABLE_NAME'].unique()

        if 'Market_Data' in market_tables or len(market_tables) > 0:
            table_name = 'Market_Data' if 'Market_Data' in market_tables else market_tables[0]
            schema_info['table_exists'] = True

            # Get sample data to understand structure
            sample_query = f"SELECT TOP 5 * FROM dbo.{table_name}"
            sample_df = run_query(sample_query)

            if not sample_df.empty:
                schema_info['available_columns'] = list(sample_df.columns)
                schema_info['sample_data'] = sample_df
                schema_info['table_name'] = table_name

    except Exception as e:
        print(f"Error checking database schema: {e}")

    return schema_info

def load_brokerage_valuation_data(years: int = 5) -> pd.DataFrame:
    """
    Load valuation metrics for brokerage sector from SQL database.
    Automatically adapts to available database schema.

    Args:
        years: Number of trailing years to include (default 5).

    Returns:
        DataFrame with columns: TICKER, TRADE_DATE, PE, PB, PS, EV_EBITDA, Type
    """

    # Check database schema first
    schema_info = check_database_schema()

    if not schema_info['table_exists']:
        st.error("❌ No market data table found in database")
        return pd.DataFrame()

    # Get broker tickers
    broker_tickers = get_broker_tickers()
    if not broker_tickers:
        st.warning("⚠️ No broker tickers found")
        return pd.DataFrame()

    # Build column list based on available columns
    available_cols = schema_info['available_columns']
    table_name = schema_info.get('table_name', 'Market_Data')

    # Required columns
    required_cols = ['TICKER', 'TRADE_DATE']
    select_cols = []

    # Map standard column names to possible variations
    column_mapping = {
        'TICKER': ['TICKER', 'SYMBOL', 'STOCK_CODE'],
        'TRADE_DATE': ['TRADE_DATE', 'DATE', 'TRADING_DATE', 'PRICE_DATE'],
        'PE': ['PE', 'P_E', 'PE_RATIO', 'PRICE_EARNINGS'],
        'PB': ['PB', 'P_B', 'PB_RATIO', 'PRICE_BOOK'],
        'PS': ['PS', 'P_S', 'PS_RATIO', 'PRICE_SALES'],
        'EV_EBITDA': ['EV_EBITDA', 'EV_TO_EBITDA', 'EVEBITDA'],
        'MKT_CAP': ['MKT_CAP', 'MARKET_CAP', 'MARKET_CAPITALIZATION'],
        'CLOSE_PRICE': ['CLOSE_PRICE', 'CLOSE', 'CLOSING_PRICE'],
        'VOLUME': ['VOLUME', 'TRADING_VOLUME']
    }

    # Find actual column names
    actual_columns = {}
    for standard_name, possible_names in column_mapping.items():
        found_col = None
        for possible in possible_names:
            if possible in available_cols:
                found_col = possible
                break
        if found_col:
            actual_columns[standard_name] = found_col
            select_cols.append(f"md.{found_col}")

    # Check if we have minimum required columns
    if 'TICKER' not in actual_columns or 'TRADE_DATE' not in actual_columns:
        st.error("❌ Essential columns (TICKER, TRADE_DATE) not found in database")
        return pd.DataFrame()

    # Build the SQL query
    ticker_placeholders = ', '.join([f"'{ticker}'" for ticker in broker_tickers])

    query = f"""
        SELECT {', '.join(select_cols)}
        FROM dbo.{table_name} AS md
        WHERE md.{actual_columns['TICKER']} IN ({ticker_placeholders})
          AND md.{actual_columns['TRADE_DATE']} >= DATEADD(year, -{years}, CAST(GETDATE() AS date))
        ORDER BY md.{actual_columns['TICKER']}, md.{actual_columns['TRADE_DATE']}
    """

    df = _load_dataframe_from_db(query)

    if df.empty:
        st.warning("⚠️ No broker valuation data found in the specified time period")
        return df

    # Rename columns to standard names
    rename_map = {v: k for k, v in actual_columns.items()}
    df = df.rename(columns=rename_map)

    # Convert TRADE_DATE to datetime
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])

    # Add broker type classification
    def get_broker_type(ticker):
        # Basic classification: 3-letter tickers are typically listed
        if len(str(ticker)) == 3 and str(ticker).isalpha():
            return 'Listed'
        else:
            return 'Unlisted'

    df['Type'] = df['TICKER'].apply(get_broker_type)

    # Add info about loaded data
    st.info(f"✅ Loaded {len(df)} records for {df['TICKER'].nunique()} brokers from {df['TRADE_DATE'].min().strftime('%Y-%m-%d')} to {df['TRADE_DATE'].max().strftime('%Y-%m-%d')}")

    return df

def load_brokerage_valuation_universe(years: int = 5) -> pd.DataFrame:
    """
    Load valuation metrics for all brokerage tickers with additional metadata.
    This is an alias for load_brokerage_valuation_data for compatibility.

    Args:
        years: Number of trailing years to include (default 5).

    Returns:
        DataFrame with valuation ratios and broker classifications.
    """
    return load_brokerage_valuation_data(years)

def get_listed_brokers() -> list:
    """Get list of listed brokers (3-letter tickers)."""
    all_tickers = get_broker_tickers()
    return [ticker for ticker in all_tickers
            if len(str(ticker)) == 3 and str(ticker).isalpha()]

def get_unlisted_brokers() -> list:
    """Get list of unlisted brokers (non-3-letter tickers)."""
    all_tickers = get_broker_tickers()
    return [ticker for ticker in all_tickers
            if not (len(str(ticker)) == 3 and str(ticker).isalpha())]

def load_brokerage_sector_map() -> pd.DataFrame:
    """Get sector mapping for brokers (Listed vs Unlisted)."""
    all_tickers = get_broker_tickers()

    data = []
    for ticker in all_tickers:
        broker_type = 'Listed' if len(str(ticker)) == 3 and str(ticker).isalpha() else 'Unlisted'
        data.append({
            'TICKER': ticker,
            'Type': broker_type,
            'Sector': 'Brokerage',
            'Listed': broker_type == 'Listed'
        })

    return pd.DataFrame(data)

@lru_cache(maxsize=1)
def _get_broker_classifications() -> dict:
    """Get broker classifications from database or static list."""
    classifications = {}

    # Get all tickers and classify them
    for ticker in get_broker_tickers():
        if len(str(ticker)) == 3 and str(ticker).isalpha():
            classifications[ticker] = 'Listed'
        else:
            classifications[ticker] = 'Unlisted'

    return classifications