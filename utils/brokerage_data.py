"""
Brokerage financial data loading from SQL database.
Replaces CSV-based data loading with live database queries.
"""

import pandas as pd
import streamlit as st
from typing import Optional, List
from utils.db import run_query

# ============================================================================
# EXCLUDED BROKERS (defunct/inactive/small brokers to filter out)
# ============================================================================

EXCLUDED_TICKERS = [
    'ABSC', 'APSC', 'ASIAS', 'ATSC', 'AVS', 'BMSC', 'CBVS', 'CLS', 'CVS',
    'CBVS', 'DDSC', 'DNSC', 'DNSE', 'DTDS', 'ECCS', 'EPS', 'FLCS', 'GBS',
    'GLS', 'GLSC', 'HASC', 'HBBS', 'HBSC', 'HPC', 'HSSC', 'HRSC', 'HVS',
    'HVSC', 'IRSC', 'ISC', 'JSI', 'JISC', 'JSIC', 'KLS', 'KEVS', 'LVSC',
    'MHBS', 'MKSC', 'MSBS', 'MSGS', 'NASC', 'NAVS', 'NHSV', 'NSIC', 'NVSC',
    'OCSC', 'PBSV', 'PCSC', 'PGSC', 'ROSE', 'RUBSE', 'SJCS', 'SME', 'STSC',
    'SVS', 'TAS', 'TCSC', 'TFSC', 'TLSC', 'TSSC', 'VCSC', 'VDSE', 'VFSC',
    'VGSC', 'VIETS', 'VNIS', 'VNSC', 'VQSC', 'VSEC', 'VSM', 'VSMC',
    'VTSC'
]

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_brokerage_metrics(
    ticker: Optional[str] = None,
    start_year: int = 2017,
    end_year: Optional[int] = None,
    include_annual: bool = False
) -> pd.DataFrame:
    """
    Load brokerage financial metrics from database.
    Returns DataFrame matching Combined_Financial_Data.csv structure.

    Args:
        ticker: Specific broker ticker (None = all brokers)
        start_year: Start year (default 2017)
        end_year: End year (None = current year)
        include_annual: Include annual data (LENGTHREPORT=5)

    Returns:
        DataFrame with columns: TICKER, YEARREPORT, LENGTHREPORT, STARTDATE, ENDDATE,
                                NOTE, STATEMENT_TYPE, METRIC_CODE, VALUE, KEYCODE,
                                KEYCODE_NAME, QUARTER_LABEL
    """

    # Build WHERE conditions
    where_conditions = [f"YEARREPORT >= {start_year}"]

    if end_year:
        where_conditions.append(f"YEARREPORT <= {end_year}")

    if ticker:
        where_conditions.append(f"TICKER = '{ticker}'")

    if not include_annual:
        where_conditions.append("LENGTHREPORT BETWEEN 1 AND 4")  # Q1-Q4 only

    # Exclude defunct/inactive brokers (but keep Sector aggregate)
    excluded_list = ','.join([f"'{t}'" for t in EXCLUDED_TICKERS])
    where_conditions.append(f"TICKER NOT IN ({excluded_list})")

    where_clause = " AND ".join(where_conditions)

    # Query database
    query = f"""
    SELECT
        TICKER,
        YEARREPORT,
        LENGTHREPORT,
        STARTDATE,
        ENDDATE,
        'Database' as NOTE,
        CASE
            WHEN KEYCODE LIKE 'BS.%' THEN 'BS'
            WHEN KEYCODE LIKE 'IS.%' THEN 'IS'
            WHEN KEYCODE LIKE '7.%' OR KEYCODE LIKE '8.%' THEN 'NOTE'
            ELSE 'CALC'
        END as STATEMENT_TYPE,
        KEYCODE as METRIC_CODE,
        VALUE,
        KEYCODE,
        KEYCODE_NAME,
        QUARTER_LABEL
    FROM dbo.BrokerageMetrics
    WHERE {where_clause}
    ORDER BY TICKER, YEARREPORT, LENGTHREPORT, KEYCODE
    """

    df = run_query(query)

    if df.empty:
        st.warning(f"⚠️ No brokerage data found for the specified filters")
        return pd.DataFrame()

    # Convert data types
    df['YEARREPORT'] = df['YEARREPORT'].astype(int)
    df['LENGTHREPORT'] = df['LENGTHREPORT'].astype(int)
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    # Convert dates
    if 'STARTDATE' in df.columns:
        df['STARTDATE'] = pd.to_datetime(df['STARTDATE'], errors='coerce')
    if 'ENDDATE' in df.columns:
        df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')

    # NOTE: METRIC_CODE is already set to KEYCODE from database query (line 87)
    # Database stores correct KEYCODEs: 'Total_Operating_Income', 'Net_Brokerage_Income', 'PBT', 'ROE', etc.
    # No translation needed!

    st.success(f"✅ Loaded {len(df):,} records for {df['TICKER'].nunique()} brokers from database")

    return df

def get_calc_metric_value(
    df: pd.DataFrame,
    ticker: str,
    year: int,
    quarter: int,
    metric_code: str
) -> float:
    """
    Get a specific metric value from database.
    All metrics including ratios (ROE, ROA, etc.) are now stored in the database.

    Args:
        df: DataFrame from load_brokerage_metrics() (can be None, will query directly)
        ticker: Broker ticker
        year: Year
        quarter: Quarter (1-4)
        metric_code: Database KEYCODE (e.g., 'Total_Operating_Income', 'PBT', 'ROE')

    Returns:
        Metric value (float)
    """

    # metric_code is already the database KEYCODE - no translation needed!

    # If df provided, use it (faster)
    if df is not None and not df.empty:
        result = df[
            (df['TICKER'] == ticker) &
            (df['YEARREPORT'] == year) &
            (df['LENGTHREPORT'] == quarter) &
            (df['METRIC_CODE'] == metric_code)
        ]

        if not result.empty:
            return float(result.iloc[0]['VALUE'])

    # Otherwise query database directly
    query = """
    SELECT VALUE
    FROM dbo.BrokerageMetrics
    WHERE TICKER = :ticker
      AND YEARREPORT = :year
      AND LENGTHREPORT = :quarter
      AND KEYCODE = :keycode
    """

    result = run_query(query, {
        'ticker': ticker,
        'year': year,
        'quarter': quarter,
        'keycode': metric_code
    })

    if not result.empty:
        return float(result.iloc[0]['VALUE'])

    return 0.0

@st.cache_data(ttl=3600)
def get_available_tickers() -> List[str]:
    """Get list of available broker tickers from database (2017 onwards)."""

    try:
        # Build exclusion list
        excluded_list = ','.join([f"'{t}'" for t in EXCLUDED_TICKERS])

        query = f"""
        SELECT DISTINCT TICKER
        FROM dbo.BrokerageMetrics
        WHERE YEARREPORT >= 2017
          AND LENGTHREPORT BETWEEN 1 AND 4
          AND TICKER NOT IN ({excluded_list})
        ORDER BY TICKER
        """

        df = run_query(query)

        if not df.empty:
            return df['TICKER'].tolist()
    except Exception as e:
        st.warning(f"⚠️ Database connection issue, using fallback ticker list. Error: {e}")

    # Fallback to default list (active brokers only)
    return ['SSI', 'VCI', 'HCM', 'VIX', 'VND', 'MBS', 'SHS', 'BSI',
            'TCBS', 'FPTS', 'AGR', 'CTS', 'VDS', 'APS', 'ORS', 'PSI',
            'BVS', 'IVS', 'EVS', 'VFS', 'VIG', 'TVS', 'CVT', 'AAS']

@st.cache_data(ttl=3600)
def get_available_quarters(ticker: Optional[str] = None) -> List[str]:
    """Get list of available quarters from database (2017 onwards), sorted newest first."""

    where_clause = "YEARREPORT >= 2017 AND LENGTHREPORT BETWEEN 1 AND 4"
    if ticker:
        where_clause += f" AND TICKER = '{ticker}'"

    query = f"""
    SELECT DISTINCT QUARTER_LABEL
    FROM dbo.BrokerageMetrics
    WHERE {where_clause}
      AND QUARTER_LABEL IS NOT NULL
      AND QUARTER_LABEL != 'Annual'
    ORDER BY QUARTER_LABEL DESC
    """

    df = run_query(query)

    if not df.empty:
        # Parse and sort quarters properly
        quarters = df['QUARTER_LABEL'].tolist()

        def quarter_sort_key(q):
            try:
                if 'Q' in q:
                    parts = q.split('Q')
                    quarter_num = int(parts[0])
                    year = int(parts[1])
                    if year < 50:
                        year += 2000
                    elif year < 100:
                        year += 1900
                    return (year, quarter_num)
            except:
                pass
            return (0, 0)

        return sorted(quarters, key=quarter_sort_key, reverse=True)

    return []

def parse_quarter_label(quarter_label: str) -> tuple:
    """Parse quarter label like '1Q24' to (year, quarter_num)."""
    try:
        if not isinstance(quarter_label, str) or 'Q' not in quarter_label:
            return (None, None)

        parts = quarter_label.split('Q')
        if len(parts) != 2:
            return (None, None)

        quarter_num = int(parts[0])
        year_str = parts[1]

        # Handle 2-digit years
        if len(year_str) == 2:
            year_int = int(year_str)
            year = 2000 + year_int if year_int <= 50 else 1900 + year_int
        else:
            year = int(year_str)

        if quarter_num not in [1, 2, 3, 4] or year < 1990 or year > 2050:
            return (None, None)

        return (year, quarter_num)

    except:
        return (None, None)

# ============================================================================
# OPTIMIZED FUNCTIONS (for performance)
# ============================================================================

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_available_tickers_fast() -> List[str]:
    """Get list of available tickers - ultra lightweight query (alias for get_available_tickers)"""
    return get_available_tickers()

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_available_years_fast() -> List[int]:
    """Get available years - ultra lightweight"""
    query = """
    SELECT DISTINCT YEARREPORT
    FROM dbo.BrokerageMetrics
    WHERE YEARREPORT >= 2017
    ORDER BY YEARREPORT DESC
    """
    df = run_query(query)
    return df['YEARREPORT'].tolist() if not df.empty else []

@st.cache_data(ttl=3600, show_spinner="Loading selected data...")
def load_filtered_brokerage_data(
    tickers: List[str],
    metrics: List[str],
    years: List[int],
    quarters: List[int]
) -> pd.DataFrame:
    """
    Load ONLY the data needed for current selections.
    Much faster than loading everything.

    Args:
        tickers: List of broker tickers to load (e.g., ['SSI', 'VCI'])
        metrics: List of database KEYCODEs to load (e.g., ['Net_Brokerage_Income', 'PBT', 'ROE'])
        years: List of years to load (e.g., [2024, 2025])
        quarters: List of quarters to load (e.g., [1, 2, 3, 4])

    Returns:
        Filtered DataFrame with only requested data
    """

    if not tickers or not years or not quarters or not metrics:
        return pd.DataFrame()

    # metrics are already database KEYCODEs - no translation needed!
    # All metrics including ratios are now stored in database with exact KEYCODEs

    # Build optimized query
    ticker_list = ','.join([f"'{t}'" for t in tickers])
    year_list = ','.join(map(str, years))
    quarter_list = ','.join(map(str, quarters))
    keycode_list = ','.join([f"'{k}'" for k in metrics])
    excluded_list = ','.join([f"'{t}'" for t in EXCLUDED_TICKERS])

    query = f"""
    SELECT
        TICKER,
        YEARREPORT,
        LENGTHREPORT,
        QUARTER_LABEL,
        KEYCODE as METRIC_CODE,
        VALUE,
        KEYCODE,
        KEYCODE_NAME,
        CASE
            WHEN KEYCODE LIKE 'BS.%' THEN 'BS'
            WHEN KEYCODE LIKE 'IS.%' THEN 'IS'
            WHEN KEYCODE LIKE '7.%' OR KEYCODE LIKE '8.%' THEN 'NOTE'
            ELSE 'CALC'
        END as STATEMENT_TYPE
    FROM dbo.BrokerageMetrics
    WHERE TICKER IN ({ticker_list})
      AND YEARREPORT IN ({year_list})
      AND LENGTHREPORT IN ({quarter_list})
      AND KEYCODE IN ({keycode_list})
      AND TICKER NOT IN ({excluded_list})
    ORDER BY TICKER, YEARREPORT, LENGTHREPORT, KEYCODE
    """

    df = run_query(query)

    if df.empty:
        return pd.DataFrame()

    # Convert data types
    df['YEARREPORT'] = df['YEARREPORT'].astype(int)
    df['LENGTHREPORT'] = df['LENGTHREPORT'].astype(int)
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    # NOTE: METRIC_CODE is already set to KEYCODE from database query (line 353)
    # Database stores correct KEYCODEs: 'Total_Operating_Income', 'Net_Brokerage_Income', 'PBT', 'ROE', etc.
    # No translation needed!

    return df

@st.cache_data(ttl=3600)
def load_ticker_quarter_data(ticker: str, quarter_label: str, lookback_quarters: int = 6) -> pd.DataFrame:
    """
    Load data for a specific ticker and quarter, with optional lookback periods.
    Optimized for AI Commentary page use case.

    Args:
        ticker: Single broker ticker (e.g., 'SSI')
        quarter_label: Quarter label (e.g., '1Q24')
        lookback_quarters: Number of historical quarters to include (default 6)

    Returns:
        DataFrame with ticker data for the specified quarter and lookback period
    """
    from utils.brokerage_data import parse_quarter_label

    # Parse the quarter label
    year, quarter_num = parse_quarter_label(quarter_label)
    if year is None or quarter_num is None:
        return pd.DataFrame()

    # Calculate lookback range (approximate - we'll get extra data and filter in Python)
    # Go back 2 years to ensure we capture enough quarters
    start_year = year - 2
    end_year = year

    excluded_list = ','.join([f"'{t}'" for t in EXCLUDED_TICKERS])

    query = f"""
    SELECT
        TICKER,
        YEARREPORT,
        LENGTHREPORT,
        QUARTER_LABEL,
        STARTDATE,
        ENDDATE,
        KEYCODE as METRIC_CODE,
        VALUE,
        KEYCODE,
        KEYCODE_NAME,
        CASE
            WHEN KEYCODE LIKE 'BS.%' THEN 'BS'
            WHEN KEYCODE LIKE 'IS.%' THEN 'IS'
            WHEN KEYCODE LIKE '7.%' OR KEYCODE LIKE '8.%' THEN 'NOTE'
            ELSE 'CALC'
        END as STATEMENT_TYPE
    FROM dbo.BrokerageMetrics
    WHERE TICKER = '{ticker}'
      AND YEARREPORT BETWEEN {start_year} AND {end_year}
      AND LENGTHREPORT BETWEEN 1 AND 4
      AND TICKER NOT IN ({excluded_list})
    ORDER BY YEARREPORT, LENGTHREPORT, KEYCODE
    """

    df = run_query(query)

    if df.empty:
        return pd.DataFrame()

    # Convert data types
    df['YEARREPORT'] = df['YEARREPORT'].astype(int)
    df['LENGTHREPORT'] = df['LENGTHREPORT'].astype(int)
    df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')

    # Convert dates if present
    if 'STARTDATE' in df.columns:
        df['STARTDATE'] = pd.to_datetime(df['STARTDATE'], errors='coerce')
    if 'ENDDATE' in df.columns:
        df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')

    # NOTE: METRIC_CODE is already set to KEYCODE from database query (line 426)
    # Database stores correct KEYCODEs: 'Total_Operating_Income', 'Net_Brokerage_Income', 'PBT', 'ROE', etc.
    # No translation needed!

    return df

@st.cache_data(ttl=3600)
def get_ticker_quarters_list(ticker: str, start_year: int = 2017) -> List[str]:
    """
    Get list of available quarters for a specific ticker.
    Lightweight query - only returns quarter labels.

    Args:
        ticker: Broker ticker
        start_year: Start year (default 2017)

    Returns:
        List of quarter labels sorted chronologically (newest first)
    """
    try:
        excluded_list = ','.join([f"'{t}'" for t in EXCLUDED_TICKERS])

        query = f"""
        SELECT DISTINCT QUARTER_LABEL, YEARREPORT, LENGTHREPORT
        FROM dbo.BrokerageMetrics
        WHERE TICKER = '{ticker}'
          AND YEARREPORT >= {start_year}
          AND LENGTHREPORT BETWEEN 1 AND 4
          AND QUARTER_LABEL IS NOT NULL
          AND QUARTER_LABEL != 'Annual'
          AND TICKER NOT IN ({excluded_list})
        ORDER BY YEARREPORT DESC, LENGTHREPORT DESC
        """

        df = run_query(query)

        if not df.empty:
            return df['QUARTER_LABEL'].tolist()
    except Exception as e:
        st.error(f"⚠️ Database error loading quarters for {ticker}: {e}")

    return []
