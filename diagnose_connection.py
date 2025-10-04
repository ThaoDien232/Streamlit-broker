#!/usr/bin/env python3
"""
Diagnose database connection issues without using Streamlit context
"""

import toml
import pyodbc
from sqlalchemy import create_engine
import urllib.parse

def diagnose_connection():
    try:
        # Load secrets
        secrets = toml.load('.streamlit/secrets.toml')
        db_config = secrets['db']

        print("=== DATABASE CONNECTION DIAGNOSIS ===")
        print(f"Available ODBC drivers: {pyodbc.drivers()}")

        print(f"\nConfig from secrets:")
        print(f"  Server: {db_config.get('server', 'Not set')}")
        print(f"  Database: {db_config.get('database', 'Not set')}")
        print(f"  Driver: {db_config.get('driver', 'Not set')}")
        print(f"  Port: {db_config.get('port', 'Not set')}")

        # Check the connection URL format
        connection_url = db_config.get('url', '')
        print(f"\nConnection URL (first 80 chars): {connection_url[:80]}...")

        # Common issues and fixes
        print(f"\n=== DIAGNOSIS ===")

        # Check if driver name matches available drivers
        configured_driver = db_config.get('driver', '')
        available_drivers = pyodbc.drivers()

        if configured_driver not in available_drivers:
            print(f"❌ Driver mismatch!")
            print(f"   Configured: '{configured_driver}'")
            print(f"   Available: {available_drivers}")
            print(f"   Suggestion: Use 'SQL Server' instead")

            # Suggest corrected connection string
            server = db_config.get('server', '')
            database = db_config.get('database', '')
            username = db_config.get('username', '')
            password = db_config.get('password', '')

            # Method 1: Direct pyodbc connection string
            direct_conn_str = f"DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
            print(f"\n🔧 Try this connection string format:")
            print(f"   Direct: {direct_conn_str}")

            # Method 2: SQLAlchemy URL
            encoded_password = urllib.parse.quote_plus(password)
            sqlalchemy_url = f"mssql+pyodbc://{username}:{encoded_password}@{server}/{database}?driver=SQL+Server"
            print(f"   SQLAlchemy: {sqlalchemy_url}")

        else:
            print(f"✓ Driver '{configured_driver}' is available")

        # Test direct pyodbc connection
        print(f"\n=== TESTING DIRECT CONNECTION ===")
        try:
            conn_str = f"DRIVER={{SQL Server}};SERVER={db_config['server']};DATABASE={db_config['database']};UID={db_config['username']};PWD={db_config['password']}"
            conn = pyodbc.connect(conn_str, timeout=10)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"✓ Direct pyodbc connection successful! Result: {result[0]}")
            conn.close()

        except Exception as e:
            print(f"❌ Direct pyodbc failed: {e}")

            if "IM002" in str(e):
                print(f"   This error means ODBC driver not found")
                print(f"   Solutions:")
                print(f"   1. Install Microsoft ODBC Driver for SQL Server")
                print(f"   2. Use pymssql instead: pip install pymssql")
                print(f"   3. Update connection string driver name")
            elif "08001" in str(e):
                print(f"   This error means connection refused - check server/port")
            elif "28000" in str(e):
                print(f"   This error means authentication failed - check credentials")

    except Exception as e:
        print(f"❌ Failed to load config: {e}")

if __name__ == "__main__":
    diagnose_connection()