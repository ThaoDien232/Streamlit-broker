import streamlit as st
import pandas as pd
import pyodbc
from contextlib import contextmanager
import os

def _get_db_connection_string():
    """Get database connection string from Streamlit secrets or environment variables"""

    # Method 1: Try DC_DB_STRING from Streamlit secrets (preferred)
    if hasattr(st, 'secrets') and "DC_DB_STRING" in st.secrets:
        return st.secrets["DC_DB_STRING"]

    # Method 2: Try environment variable (fallback)
    env_conn_string = os.getenv('DC_DB_STRING')
    if env_conn_string:
        return env_conn_string

    # Method 3: Check if we're in Streamlit context but missing secrets
    if hasattr(st, 'secrets'):
        raise RuntimeError(
            "Database connection string missing from Streamlit secrets. "
            "Please add DC_DB_STRING to your Streamlit Cloud app secrets."
        )

    # Method 4: Not in Streamlit context and no environment variables
    raise RuntimeError(
        "Database connection string not found. Please set environment variable:\n"
        "DC_DB_STRING\n"
        "Or configure DC_DB_STRING in Streamlit Cloud dashboard secrets."
    )

@contextmanager
def get_connection():
    """Create database connection using pyodbc with DC_DB_STRING"""
    connection = None

    try:
        # Get connection string from available sources
        connection_string = _get_db_connection_string()

        # Validate connection string exists
        if not connection_string:
            raise RuntimeError("Database connection string (DC_DB_STRING) is empty or invalid")

        # Connect using pyodbc
        connection = pyodbc.connect(connection_string, timeout=30)

        # Yield the connection
        try:
            yield connection
        finally:
            if connection:
                connection.close()

    except Exception as e:
        if hasattr(st, 'error'):
            st.error(f"Database connection failed: {e}")
        else:
            print(f"Database connection failed: {e}")
        raise

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    try:
        with get_connection() as conn:
            # pyodbc connection - use ? placeholders for parameters
            if params:
                # Convert named parameters (:param) to positional (?) for pyodbc
                formatted_sql = sql
                param_values = []

                for key, value in params.items():
                    formatted_sql = formatted_sql.replace(f":{key}", "?")
                    param_values.append(value)

                result = pd.read_sql(formatted_sql, conn, params=param_values)
            else:
                result = pd.read_sql(sql, conn)

            return result

    except Exception as e:
        st.error(f"Database query failed: {e}")
        print(f"SQL Query error: {e}")
        print(f"Query: {sql}")
        if params:
            print(f"Parameters: {params}")
        # Return empty DataFrame instead of raising to prevent app crash
        return pd.DataFrame()

def test_connection() -> bool:
    """Test database connection and return True if successful"""
    try:
        with get_connection() as conn:
            # pyodbc connection test
            cursor = conn.cursor()
            cursor.execute("SELECT 1 as test")
            result = cursor.fetchone()
            cursor.close()
            return True
    except Exception as e:
        st.error(f"Database connection test failed: {e}")
        return False

def get_table_info(table_name: str) -> pd.DataFrame:
    """Get column information for a specific table"""
    query = """
    SELECT
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE,
        COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = :table_name
    ORDER BY ORDINAL_POSITION
    """
    return run_query(query, {"table_name": table_name})

def get_available_tables() -> pd.DataFrame:
    """Get list of available tables in the database"""
    query = """
    SELECT
        TABLE_SCHEMA,
        TABLE_NAME,
        TABLE_TYPE
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_TYPE = 'BASE TABLE'
    ORDER BY TABLE_SCHEMA, TABLE_NAME
    """
    return run_query(query)

def get_latest_valuation_data(ticker: str = None) -> pd.DataFrame:
    """Get latest valuation data (PE, PB, PS, EV_EBITDA) with OHLC prices"""
    if ticker:
        query = """
        SELECT TOP 1
            TICKER,
            TRADE_DATE,
            PE,
            PB,
            PS,
            EV_EBITDA,
            PX_OPEN,
            PX_HIGH,
            PX_LOW,
            PX_LAST,
            MKT_CAP
        FROM Market_Data
        WHERE TICKER = :ticker
        ORDER BY TRADE_DATE DESC
        """
        params = {"ticker": ticker.upper()}
    else:
        query = """
        WITH LatestDate AS (
            SELECT MAX(TRADE_DATE) as max_date FROM Market_Data
        )
        SELECT
            m.TICKER,
            m.TRADE_DATE,
            m.PE,
            m.PB,
            m.PS,
            m.EV_EBITDA,
            m.PX_OPEN,
            m.PX_HIGH,
            m.PX_LOW,
            m.PX_LAST,
            m.MKT_CAP
        FROM Market_Data m
        INNER JOIN LatestDate l ON m.TRADE_DATE = l.max_date
        ORDER BY m.TICKER
        """
        params = None

    return run_query(query, params)

def get_valuation_history(ticker: str, days: int = 30) -> pd.DataFrame:
    """Get historical valuation data for a specific ticker"""
    query = """
    SELECT TOP (:days)
        TICKER,
        TRADE_DATE,
        PE,
        PB,
        PS,
        EV_EBITDA,
        PX_LAST,
        MKT_CAP
    FROM Market_Data
    WHERE TICKER = :ticker
    ORDER BY TRADE_DATE DESC
    """
    params = {"ticker": ticker.upper(), "days": days}
    return run_query(query, params)

def get_sector_valuation_comparison() -> pd.DataFrame:
    """Get latest valuation metrics by sector for comparison"""
    query = """
    WITH LatestDate AS (
        SELECT MAX(TRADE_DATE) as max_date FROM Market_Data
    )
    SELECT
        s.Sector,
        s.L1 as Industry,
        m.TICKER,
        m.PE,
        m.PB,
        m.PS,
        m.EV_EBITDA,
        m.MKT_CAP,
        s.VNI as VN30_Member
    FROM Market_Data m
    INNER JOIN LatestDate l ON m.TRADE_DATE = l.max_date
    INNER JOIN Sector_Map s ON m.TICKER = s.Ticker
    WHERE m.PE IS NOT NULL
    ORDER BY s.Sector, m.PE
    """
    return run_query(query)

def get_vn30_valuation() -> pd.DataFrame:
    """Get valuation metrics for VN30 index constituents"""
    query = """
    WITH LatestDate AS (
        SELECT MAX(TRADE_DATE) as max_date FROM Market_Data
    )
    SELECT
        m.TICKER,
        s.L1 as Industry,
        m.PE,
        m.PB,
        m.PS,
        m.EV_EBITDA,
        m.PX_LAST as Price,
        m.MKT_CAP
    FROM Market_Data m
    INNER JOIN LatestDate l ON m.TRADE_DATE = l.max_date
    INNER JOIN Sector_Map s ON m.TICKER = s.Ticker
    WHERE s.VNI = 'Y'
    ORDER BY m.MKT_CAP DESC
    """
    return run_query(query)

def get_valuation_screening(
    pe_min: float = None, pe_max: float = None,
    pb_min: float = None, pb_max: float = None,
    ps_min: float = None, ps_max: float = None,
    sector: str = None
) -> pd.DataFrame:
    """Screen stocks based on valuation criteria"""
    where_conditions = ["m.PE IS NOT NULL"]
    params = {}

    if pe_min is not None:
        where_conditions.append("m.PE >= :pe_min")
        params["pe_min"] = pe_min
    if pe_max is not None:
        where_conditions.append("m.PE <= :pe_max")
        params["pe_max"] = pe_max
    if pb_min is not None:
        where_conditions.append("m.PB >= :pb_min")
        params["pb_min"] = pb_min
    if pb_max is not None:
        where_conditions.append("m.PB <= :pb_max")
        params["pb_max"] = pb_max
    if ps_min is not None:
        where_conditions.append("m.PS >= :ps_min")
        params["ps_min"] = ps_min
    if ps_max is not None:
        where_conditions.append("m.PS <= :ps_max")
        params["ps_max"] = ps_max
    if sector:
        where_conditions.append("s.Sector = :sector")
        params["sector"] = sector

    where_clause = " AND ".join(where_conditions)

    query = f"""
    WITH LatestDate AS (
        SELECT MAX(TRADE_DATE) as max_date FROM Market_Data
    )
    SELECT
        m.TICKER,
        s.Sector,
        s.L1 as Industry,
        m.PE,
        m.PB,
        m.PS,
        m.EV_EBITDA,
        m.PX_LAST as Price,
        m.MKT_CAP,
        s.VNI as VN30_Member
    FROM Market_Data m
    INNER JOIN LatestDate l ON m.TRADE_DATE = l.max_date
    INNER JOIN Sector_Map s ON m.TICKER = s.Ticker
    WHERE {where_clause}
    ORDER BY m.PE
    """

    return run_query(query, params)

def get_historical_prices(ticker: str, start_date: str = '2020-01-01') -> pd.DataFrame:
    """
    Get historical closing prices for a specific ticker from dclab Market_Data table.
    Returns DataFrame with 'tradingDate' and 'close' columns.
    Note: Database stores prices in VND, we divide by 1000 to match VCI format (thousands VND).
    """
    query = """
    SELECT
        TRADE_DATE as tradingDate,
        PX_LAST as [close]
    FROM Market_Data
    WHERE TICKER = :ticker
      AND TRADE_DATE >= :start_date
      AND PX_LAST IS NOT NULL
    ORDER BY TRADE_DATE ASC
    """
    params = {"ticker": ticker.upper(), "start_date": start_date}

    df = run_query(query, params)

    if not df.empty:
        # Convert tradingDate to datetime
        df['tradingDate'] = pd.to_datetime(df['tradingDate'])

        # Convert prices from VND to thousands to match VCI format (divide by 1000)
        df['close'] = df['close'] / 1000

    return df

def get_latest_price(ticker: str) -> float:
    """Get the most recent closing price for a ticker from Market_Data table"""
    query = """
    SELECT TOP 1
        PX_LAST as close
    FROM Market_Data
    WHERE TICKER = :ticker
    ORDER BY TRADE_DATE DESC
    """
    params = {"ticker": ticker.upper()}

    df = run_query(query, params)

    if not df.empty:
        return df.iloc[0]['close'] / 1000  # Convert to thousands
    return None
