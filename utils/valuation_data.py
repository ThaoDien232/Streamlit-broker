"""
Data loading utilities for brokerage sector valuation analysis.
Uses the same SQL database as banking sector but filtered for brokerage tickers.
"""

from __future__ import annotations
from functools import lru_cache
from typing import Optional
import pandas as pd
from utils.db import run_query

def _load_dataframe_from_db(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    """Load dataframe using existing database connection."""
    try:
        return run_query(query, params)
    except Exception as e:
        import streamlit as st
        st.error(f"Database connection failed: {e}")
        print(f"Database query failed: {e}")
        return pd.DataFrame()

def load_brokerage_valuation_data(years: int = 5) -> pd.DataFrame:
    """
    Load valuation metrics for brokerage sector from SQL database.
    Uses the same structure as banking valuation but filters for broker tickers.

    Args:
        years: Number of trailing years to include (default 5).

    Returns:
        DataFrame with columns: TICKER, TRADE_DATE, PE, PB, PS, EV_EBITDA, Type
    """

    # Define broker tickers (you can expand this list)
    broker_tickers = [
        'SSI', 'VCI', 'HCM', 'VIX', 'VND', 'MBS', 'SHS', 'BSI',
        'TCBS', 'FPTS', 'AGR', 'CTS', 'VDS', 'APS', 'ORS', 'PSI',
        'BVS', 'IVS', 'EVS', 'VFS', 'VIG', 'TVS', 'CVT', 'AAS',
        'WSS', 'TCI', 'DSC', 'VRS', 'VTS', 'VSC', 'SHB', 'TGG'
    ]

    # Create the ticker list for SQL IN clause
    ticker_placeholders = ', '.join([f"'{ticker}'" for ticker in broker_tickers])

    query = f"""
        SELECT md.TICKER,
               md.TRADE_DATE,
               md.PE,
               md.PB,
               md.PS,
               md.EV_EBITDA,
               md.MKT_CAP,
               'Listed' AS Type
        FROM dbo.Market_Data AS md
        WHERE md.TICKER IN ({ticker_placeholders})
          AND md.TRADE_DATE >= DATEADD(year, -{years}, CAST(GETDATE() AS date))
          AND (md.PE IS NOT NULL OR md.PB IS NOT NULL)
        ORDER BY md.TICKER, md.TRADE_DATE
    """

    df = _load_dataframe_from_db(query)

    if df.empty:
        return df

    # Convert TRADE_DATE to datetime
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])

    # Add broker type classification (Listed vs Unlisted)
    # For now, assume all are Listed since they're in Market_Data
    # You can enhance this with a Broker_Classification table later
    def get_broker_type(ticker):
        # Basic classification: 3-letter tickers are typically listed
        if len(str(ticker)) == 3 and str(ticker).isalpha():
            return 'Listed'
        else:
            return 'Unlisted'

    df['Type'] = df['TICKER'].apply(get_broker_type)

    return df

def load_brokerage_valuation_universe(years: int = 5) -> pd.DataFrame:
    """
    Load valuation metrics for all brokerage tickers with additional metadata.
    Similar to load_valuation_universe but for brokers.

    Args:
        years: Number of trailing years to include (default 5).

    Returns:
        DataFrame with valuation ratios and broker classifications.
    """

    if years <= 0:
        raise ValueError("years must be positive")

    # Define broker tickers
    broker_tickers = [
        'SSI', 'VCI', 'HCM', 'VIX', 'VND', 'MBS', 'SHS', 'BSI',
        'TCBS', 'FPTS', 'AGR', 'CTS', 'VDS', 'APS', 'ORS', 'PSI',
        'BVS', 'IVS', 'EVS', 'VFS', 'VIG', 'TVS', 'CVT', 'AAS',
        'WSS', 'TCI', 'DSC', 'VRS', 'VTS', 'VSC', 'SHB', 'TGG'
    ]

    ticker_placeholders = ', '.join([f"'{ticker}'" for ticker in broker_tickers])

    query = f"""
        SELECT md.TICKER,
               md.TRADE_DATE,
               md.PE,
               md.PB,
               md.PS,
               md.EV_EBITDA,
               md.MKT_CAP,
               'Brokerage' AS Sector,
               CASE
                   WHEN LEN(md.TICKER) = 3 THEN 'Listed'
                   ELSE 'Unlisted'
               END AS Broker_Type,
               CASE
                   WHEN md.MKT_CAP >= 10000000000000 THEN 'Large'
                   WHEN md.MKT_CAP >= 5000000000000 THEN 'Mid'
                   ELSE 'Small'
               END AS Size_Category
        FROM dbo.Market_Data AS md
        WHERE md.TICKER IN ({ticker_placeholders})
          AND md.TRADE_DATE >= DATEADD(year, -{years}, CAST(GETDATE() AS date))
          AND (md.PE IS NOT NULL
               OR md.PB IS NOT NULL
               OR md.PS IS NOT NULL
               OR md.EV_EBITDA IS NOT NULL)
        ORDER BY md.TICKER, md.TRADE_DATE
    """

    df = _load_dataframe_from_db(query)

    if df.empty:
        return df

    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])

    # Convert numeric columns
    for col in ['PE', 'PB', 'PS', 'EV_EBITDA', 'MKT_CAP']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Rename for consistency with analysis functions
    rename_map = {
        'Broker_Type': 'Type',
        'Size_Category': 'Size'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    return df

def get_brokerage_tickers() -> list:
    """Get list of all brokerage tickers from the database."""

    broker_tickers = [
        'SSI', 'VCI', 'HCM', 'VIX', 'VND', 'MBS', 'SHS', 'BSI',
        'TCBS', 'FPTS', 'AGR', 'CTS', 'VDS', 'APS', 'ORS', 'PSI',
        'BVS', 'IVS', 'EVS', 'VFS', 'VIG', 'TVS', 'CVT', 'AAS',
        'WSS', 'TCI', 'DSC', 'VRS', 'VTS', 'VSC', 'SHB', 'TGG'
    ]

    # You could also query the database to get all broker tickers:
    # query = """
    #     SELECT DISTINCT TICKER
    #     FROM dbo.Market_Data
    #     WHERE TICKER IN (SELECT TICKER FROM dbo.Broker_List)
    #     ORDER BY TICKER
    # """
    # df = _load_dataframe_from_db(query)
    # return df['TICKER'].tolist() if not df.empty else broker_tickers

    return sorted(broker_tickers)

def get_listed_brokers() -> list:
    """Get list of listed brokers (3-letter tickers)."""
    all_tickers = get_brokerage_tickers()
    return [ticker for ticker in all_tickers
            if len(str(ticker)) == 3 and str(ticker).isalpha()]

def get_unlisted_brokers() -> list:
    """Get list of unlisted brokers (non-3-letter tickers)."""
    all_tickers = get_brokerage_tickers()
    return [ticker for ticker in all_tickers
            if not (len(str(ticker)) == 3 and str(ticker).isalpha())]

def load_brokerage_sector_map() -> pd.DataFrame:
    """Get sector mapping for brokers (Listed vs Unlisted)."""

    # This could query a Broker_Classification table if it exists:
    # query = """
    #     SELECT TICKER, BROKER_TYPE as Type, 'Brokerage' as Sector
    #     FROM dbo.Broker_Classification
    #     WHERE SECTOR = 'Brokerage'
    # """

    # For now, create from the ticker list
    all_tickers = get_brokerage_tickers()

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
    for ticker in get_brokerage_tickers():
        if len(str(ticker)) == 3 and str(ticker).isalpha():
            classifications[ticker] = 'Listed'
        else:
            classifications[ticker] = 'Unlisted'

    return classifications