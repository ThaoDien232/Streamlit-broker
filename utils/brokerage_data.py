"""
Brokerage financial data loading from SQL database.
Replaces CSV-based data loading with live database queries.
"""

import pandas as pd
import streamlit as st
from typing import Optional, List
from utils.db import run_query
from utils.keycode_mapping import (
    get_db_keycode,
    get_metric_code,
    needs_calculation,
    CALCULATED_METRICS
)

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
    'VGSC', 'VIETS', 'VNIS', 'VNSC', 'VPBS', 'VQSC', 'VSEC', 'VSM', 'VSMC',
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

    # IMPORTANT: Translate KEYCODE back to our METRIC_CODE format
    from utils.keycode_mapping import get_metric_code
    df['METRIC_CODE'] = df['KEYCODE'].apply(lambda k: get_metric_code(k) or k)

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
    Get a specific calculated metric value from database or calculate it.
    Compatible with existing code that used CSV data.

    Args:
        df: DataFrame from load_brokerage_metrics() (can be None, will query directly)
        ticker: Broker ticker
        year: Year
        quarter: Quarter (1-4)
        metric_code: Our METRIC_CODE format

    Returns:
        Metric value (float)
    """

    # Check if this metric needs calculation
    if needs_calculation(metric_code):
        return calculate_metric(ticker, year, quarter, metric_code, df)

    # Translate to database KEYCODE
    db_keycode = get_db_keycode(metric_code)

    if db_keycode is None:
        st.warning(f"⚠️ Unknown metric: {metric_code}")
        return 0.0

    # If df provided, use it (faster)
    if df is not None and not df.empty:
        result = df[
            (df['TICKER'] == ticker) &
            (df['YEARREPORT'] == year) &
            (df['LENGTHREPORT'] == quarter) &
            (df['METRIC_CODE'] == db_keycode)
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
        'keycode': db_keycode
    })

    if not result.empty:
        return float(result.iloc[0]['VALUE'])

    return 0.0

def calculate_metric(
    ticker: str,
    year: int,
    quarter: int,
    metric_code: str,
    df: Optional[pd.DataFrame] = None
) -> float:
    """
    Calculate metrics that aren't in the database.

    Args:
        ticker: Broker ticker
        year: Year
        quarter: Quarter
        metric_code: Metric to calculate (ROE, ROA, INTEREST_RATE, etc.)
        df: Optional DataFrame to use (faster than querying)

    Returns:
        Calculated value
    """

    if metric_code not in CALCULATED_METRICS:
        return 0.0

    formula_info = CALCULATED_METRICS[metric_code]
    components = formula_info['components']

    # Get component values
    values = {}
    for component in components:
        values[component] = get_calc_metric_value(df, ticker, year, quarter, component)

    # Calculate based on formula
    if metric_code == 'ROE':
        # ROE = NPAT / TOTAL_EQUITY * 100
        if values['TOTAL_EQUITY'] == 0:
            return 0.0
        roe = (values['NPAT'] / values['TOTAL_EQUITY']) * 100

        # Annualize if quarterly
        if quarter in [1, 2, 3, 4] and formula_info['annualize_quarterly']:
            roe = roe * 4

        return roe

    elif metric_code == 'ROA':
        # ROA = NPAT / TOTAL_ASSETS * 100
        if values['TOTAL_ASSETS'] == 0:
            return 0.0
        roa = (values['NPAT'] / values['TOTAL_ASSETS']) * 100

        # Annualize if quarterly
        if quarter in [1, 2, 3, 4] and formula_info['annualize_quarterly']:
            roa = roa * 4

        return roa

    elif metric_code == 'INTEREST_RATE':
        # Interest Rate = INTEREST_EXPENSE / AVG_BORROWING_BALANCE
        # Need current and previous borrowing balance to calculate average
        current_borrowing = values['BORROWING_BALANCE']

        # Get previous quarter borrowing
        if quarter == 1:
            prev_year, prev_quarter = year - 1, 4
        else:
            prev_year, prev_quarter = year, quarter - 1

        prev_borrowing = get_calc_metric_value(df, ticker, prev_year, prev_quarter, 'BORROWING_BALANCE')

        avg_borrowing = (current_borrowing + prev_borrowing) / 2

        if avg_borrowing == 0:
            return 0.0

        interest_rate = values['INTEREST_EXPENSE'] / avg_borrowing

        # Annualize if quarterly
        if quarter in [1, 2, 3, 4] and formula_info['annualize_quarterly']:
            interest_rate = interest_rate * 4

        return interest_rate

    elif metric_code == 'MARGIN_EQUITY_RATIO':
        # Margin/Equity % = MARGIN_BALANCE / TOTAL_EQUITY * 100
        if values['TOTAL_EQUITY'] == 0:
            return 0.0
        return (values['MARGIN_BALANCE'] / values['TOTAL_EQUITY']) * 100

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
        metrics: List of METRIC_CODEs to load (e.g., ['NET_BROKERAGE_INCOME', 'PBT'])
        years: List of years to load (e.g., [2024, 2025])
        quarters: List of quarters to load (e.g., [1, 2, 3, 4])

    Returns:
        Filtered DataFrame with only requested data
    """

    if not tickers or not years or not quarters:
        return pd.DataFrame()

    # Separate calculated vs database metrics
    db_metrics = []
    calculated_metrics_list = []

    for metric in metrics:
        if needs_calculation(metric):
            calculated_metrics_list.append(metric)
        else:
            db_metrics.append(metric)

    # Translate database metrics to KEYCODEs
    db_keycodes = []
    for metric in db_metrics:
        keycode = get_db_keycode(metric)
        if keycode:
            db_keycodes.append(keycode)

    # For calculated metrics, we need to load their components
    for calc_metric in calculated_metrics_list:
        formula_info = CALCULATED_METRICS.get(calc_metric)
        if formula_info:
            for component in formula_info['components']:
                component_keycode = get_db_keycode(component)
                if component_keycode and component_keycode not in db_keycodes:
                    db_keycodes.append(component_keycode)

    if not db_keycodes:
        return pd.DataFrame()

    # Build optimized query
    ticker_list = ','.join([f"'{t}'" for t in tickers])
    year_list = ','.join(map(str, years))
    quarter_list = ','.join(map(str, quarters))
    keycode_list = ','.join([f"'{k}'" for k in db_keycodes])
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

    # IMPORTANT: Translate KEYCODE back to our METRIC_CODE format
    # Database returns 'Net_Brokerage_Income', 'Net_Trading_Income', etc.
    # But we need 'NET_BROKERAGE_INCOME', 'NET_TRADING_INCOME', etc.
    # to match what the Charts/Historical pages filter by
    from utils.keycode_mapping import get_metric_code
    df['METRIC_CODE'] = df['KEYCODE'].apply(lambda k: get_metric_code(k) or k)

    # Calculate requested metrics that aren't in the database (like ROE, ROA)
    if calculated_metrics_list:
        calculated_rows = []

        for ticker in tickers:
            for year in years:
                for quarter in quarters:
                    for calc_metric in calculated_metrics_list:
                        # Calculate the metric value
                        value = calculate_metric(ticker, year, quarter, calc_metric, df)

                        if value != 0:  # Only add non-zero values
                            # Find quarter label
                            quarter_label_row = df[
                                (df['TICKER'] == ticker) &
                                (df['YEARREPORT'] == year) &
                                (df['LENGTHREPORT'] == quarter)
                            ]

                            quarter_label = quarter_label_row['QUARTER_LABEL'].iloc[0] if not quarter_label_row.empty else f"{quarter}Q{year%100}"

                            calculated_rows.append({
                                'TICKER': ticker,
                                'YEARREPORT': year,
                                'LENGTHREPORT': quarter,
                                'QUARTER_LABEL': quarter_label,
                                'METRIC_CODE': calc_metric,
                                'VALUE': value,
                                'KEYCODE': calc_metric,
                                'KEYCODE_NAME': calc_metric,
                                'STATEMENT_TYPE': 'CALC'
                            })

        # Append calculated metrics to the dataframe
        if calculated_rows:
            calc_df = pd.DataFrame(calculated_rows)
            df = pd.concat([df, calc_df], ignore_index=True)

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

    # IMPORTANT: Translate KEYCODE back to our METRIC_CODE format
    from utils.keycode_mapping import get_metric_code
    df['METRIC_CODE'] = df['KEYCODE'].apply(lambda k: get_metric_code(k) or k)

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
