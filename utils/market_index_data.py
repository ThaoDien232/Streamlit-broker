"""
Market index data loading from SQL database.
Replaces INDEX.csv with live database queries from MarketIndex table.
"""

import pandas as pd
import streamlit as st
from typing import Optional
from utils.db import run_query

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_market_index(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    index_code: str = 'VNINDEX'
) -> pd.DataFrame:
    """
    Load market index data from database.
    Returns DataFrame matching INDEX.csv structure.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format (None = no filter)
        end_date: End date in 'YYYY-MM-DD' format (None = no filter)
        index_code: Index code (default 'VNINDEX')

    Returns:
        DataFrame with columns: TRADINGDATE, COMGROUPCODE, TOTALVALUE,
                                FOREIGNBUYVOLUME, FOREIGNSELLVOLUME, etc.
    """

    # Build WHERE conditions
    where_conditions = [f"COMGROUPCODE = '{index_code}'"]

    if start_date:
        where_conditions.append(f"TRADINGDATE >= '{start_date}'")

    if end_date:
        where_conditions.append(f"TRADINGDATE <= '{end_date}'")

    where_clause = " AND ".join(where_conditions)

    # Query MarketIndex table
    query = f"""
    SELECT
        TRADINGDATE,
        COMGROUPCODE,
        TOTALVALUE,
        FOREIGNBUYVOLUME,
        FOREIGNSELLVOLUME,
        FOREIGNBUYVALUE,
        FOREIGNSELLVALUE,
        INDEXVALUE,
        CHANGEINDEX,
        RATEINDEX
    FROM dbo.MarketIndex
    WHERE {where_clause}
    ORDER BY TRADINGDATE
    """

    df = run_query(query)

    if df.empty:
        st.warning(f"⚠️ No market index data found for {index_code}")
        return pd.DataFrame()

    # Convert data types
    df['TRADINGDATE'] = pd.to_datetime(df['TRADINGDATE'])
    df['TOTALVALUE'] = pd.to_numeric(df['TOTALVALUE'], errors='coerce')

    # Add year and quarter for aggregation
    df['Year'] = df['TRADINGDATE'].dt.year
    df['Quarter'] = df['TRADINGDATE'].dt.quarter

    return df

@st.cache_data(ttl=1800)
def load_market_liquidity_data(start_year: int = 2017) -> pd.DataFrame:
    """
    Load and calculate quarterly market liquidity (average daily turnover).
    Replaces the calculation from INDEX.csv.

    Args:
        start_year: Start year (default 2017)

    Returns:
        DataFrame with columns: Year, Quarter, Quarter_Label, Avg Daily Turnover (B VND)
    """

    query = f"""
    SELECT
        YEAR(TRADINGDATE) as Year,
        DATEPART(QUARTER, TRADINGDATE) as Quarter,
        AVG(TOTALVALUE) as avg_daily_turnover
    FROM dbo.MarketIndex
    WHERE COMGROUPCODE = 'VNINDEX'
      AND YEAR(TRADINGDATE) >= {start_year}
    GROUP BY YEAR(TRADINGDATE), DATEPART(QUARTER, TRADINGDATE)
    ORDER BY Year DESC, Quarter DESC
    """

    df = run_query(query)

    if df.empty:
        st.warning(f"⚠️ No market liquidity data found from {start_year} onwards")
        return pd.DataFrame()

    # Convert to billions VND
    df['Avg Daily Turnover (B VND)'] = df['avg_daily_turnover'] / 1e9

    # Create quarter label (e.g., '1Q24')
    df['Quarter_Label'] = df.apply(
        lambda row: f"{int(row['Quarter'])}Q{int(row['Year']) % 100:02d}",
        axis=1
    )

    # Select final columns
    df = df[['Year', 'Quarter', 'Quarter_Label', 'Avg Daily Turnover (B VND)']]

    return df

@st.cache_data(ttl=3600)
def get_quarterly_turnover(year: int, quarter: int) -> float:
    """
    Get average daily turnover for a specific quarter.

    Args:
        year: Year (e.g., 2024)
        quarter: Quarter (1-4)

    Returns:
        Average daily turnover in billions VND
    """

    query = """
    SELECT AVG(TOTALVALUE) as avg_daily_turnover
    FROM dbo.MarketIndex
    WHERE COMGROUPCODE = 'VNINDEX'
      AND YEAR(TRADINGDATE) = :year
      AND DATEPART(QUARTER, TRADINGDATE) = :quarter
    """

    result = run_query(query, {'year': year, 'quarter': quarter})

    if not result.empty:
        return float(result.iloc[0]['avg_daily_turnover']) / 1e9

    return 0.0

@st.cache_data(ttl=3600)
def get_market_turnover_stats(start_year: int = 2017) -> pd.DataFrame:
    """
    Get market turnover statistics by year.
    Replaces MarketTurnover table queries.

    Args:
        start_year: Start year (default 2017)

    Returns:
        DataFrame with yearly turnover statistics
    """

    query = f"""
    SELECT
        YEAR(TRADINGDATE) as Year,
        AVG(TOTALVALUE) as avg_daily_turnover,
        SUM(TOTALVALUE) as total_turnover,
        COUNT(*) as trading_days
    FROM dbo.MarketIndex
    WHERE COMGROUPCODE = 'VNINDEX'
      AND YEAR(TRADINGDATE) >= {start_year}
    GROUP BY YEAR(TRADINGDATE)
    ORDER BY Year DESC
    """

    df = run_query(query)

    if df.empty:
        return pd.DataFrame()

    # Convert to billions VND
    df['Avg Daily Turnover (B VND)'] = df['avg_daily_turnover'] / 1e9
    df['Total Turnover (T VND)'] = df['total_turnover'] / 1e12

    return df
