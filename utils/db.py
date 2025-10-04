import streamlit as st
import pandas as pd
import pymssql
import pyodbc
from contextlib import contextmanager
from sqlalchemy import create_engine, text
import urllib.parse
import os

def _get_db_config():
    """Get database configuration from Streamlit secrets or environment variables"""

    # Method 1: Try Streamlit secrets (preferred for Streamlit Cloud)
    if hasattr(st, 'secrets') and "db" in st.secrets:
        return dict(st.secrets["db"])

    # Method 2: Try environment variables (fallback for other deployments)
    env_config = {
        'server': os.getenv('DB_SERVER'),
        'database': os.getenv('DB_DATABASE'),
        'username': os.getenv('DB_USERNAME'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '1433'),
        'url': os.getenv('DB_URL')
    }

    # Check if we have the minimum required environment variables
    if env_config['server'] and env_config['database'] and env_config['username'] and env_config['password']:
        return env_config

    # Method 3: Check if we're in Streamlit context but missing secrets
    if hasattr(st, 'secrets'):
        raise RuntimeError(
            "Database configuration missing from Streamlit secrets. "
            "Please add [db] section to your Streamlit Cloud app secrets."
        )

    # Method 4: Not in Streamlit context and no environment variables
    raise RuntimeError(
        "Database configuration not found. Please set environment variables:\n"
        "DB_SERVER, DB_DATABASE, DB_USERNAME, DB_PASSWORD\n"
        "Or configure secrets in Streamlit Cloud dashboard."
    )

@contextmanager
def get_connection():
    """Create database connection with multiple configuration sources"""
    connection = None

    try:
        # Get database configuration from available sources
        db_config = _get_db_config()

        # Validate required database configuration
        required_keys = ['server', 'database', 'username', 'password']
        missing_keys = [key for key in required_keys if not db_config.get(key)]
        if missing_keys:
            raise RuntimeError(f"Missing required database configuration: {missing_keys}")

        # Method 1: Try pyodbc with SQL Server driver (most compatible with Azure SQL)
        try:
            # Build ODBC connection string for Azure SQL
            connection_string = (
                f"DRIVER={{SQL Server}};"
                f"SERVER={db_config['server']};"
                f"DATABASE={db_config['database']};"
                f"UID={db_config['username']};"
                f"PWD={db_config['password']};"
                f"Encrypt=yes;"
                f"TrustServerCertificate=no;"
                f"Connection Timeout=30;"
            )

            connection = pyodbc.connect(connection_string, timeout=30)

        except Exception as pyodbc_error:
            # Method 2: Fallback to pymssql (for non-Azure SQL Server)
            try:
                server = db_config['server']
                port = int(db_config.get('port', 1433))

                # Handle server,port format
                if ',' in server:
                    host_part, port_part = server.split(',', 1)
                    server = host_part
                    try:
                        port = int(port_part)
                    except ValueError:
                        port = 1433

                connection = pymssql.connect(
                    server=server,
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database'],
                    port=port,
                    charset='UTF-8',
                    autocommit=False,
                    timeout=30,
                    login_timeout=30
                )

            except Exception as pymssql_error:
                # Method 3: Last resort - SQLAlchemy with URL (if available)
                try:
                    if not db_config.get("url"):
                        raise RuntimeError("No 'url' found in database configuration")

                    connection_url = db_config["url"]
                    # Fix driver name in URL
                    fixed_url = connection_url.replace("ODBC+Driver+18+for+SQL+Server", "SQL+Server")

                    engine = create_engine(fixed_url, pool_pre_ping=True)
                    connection = engine.connect()

                except Exception as sqlalchemy_error:
                    raise RuntimeError(
                        f"All database connection methods failed. Please check your database configuration.\n"
                        f"Errors:\n"
                        f"  pyodbc: {pyodbc_error}\n"
                        f"  pymssql: {pymssql_error}\n"
                        f"  sqlalchemy: {sqlalchemy_error}"
                    )

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
            # Detect connection type and handle accordingly
            connection_type = type(conn).__name__

            if connection_type == 'Connection' and hasattr(conn, 'execute'):
                # SQLAlchemy connection
                if params:
                    # Use SQLAlchemy text() for named parameters
                    result = pd.read_sql(text(sql), conn, params=params)
                else:
                    result = pd.read_sql(text(sql), conn)

            elif hasattr(conn, 'cursor'):
                # pyodbc or pymssql connection
                if params:
                    # Convert named parameters (:param) to positional (%s or ?)
                    formatted_sql = sql
                    param_values = []

                    # Determine parameter style based on connection
                    if 'pyodbc' in str(type(conn)):
                        # pyodbc uses ? placeholders
                        for key, value in params.items():
                            formatted_sql = formatted_sql.replace(f":{key}", "?")
                            param_values.append(value)
                    else:
                        # pymssql uses %s placeholders
                        for key, value in params.items():
                            formatted_sql = formatted_sql.replace(f":{key}", "%s")
                            param_values.append(value)

                    result = pd.read_sql(formatted_sql, conn, params=param_values)
                else:
                    result = pd.read_sql(sql, conn)
            else:
                # Direct pandas read_sql
                result = pd.read_sql(sql, conn, params=params)

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
            # Handle different connection types
            connection_type = type(conn).__name__

            if connection_type == 'Connection' and hasattr(conn, 'execute'):
                # SQLAlchemy connection
                result = conn.execute(text("SELECT 1 as test"))
                result.fetchone()
            elif hasattr(conn, 'cursor'):
                # pyodbc or pymssql connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                cursor.close()
            else:
                # Try direct execution
                result = conn.execute("SELECT 1 as test")
                result.fetchone()

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
