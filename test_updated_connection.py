#!/usr/bin/env python3
"""
Test the updated db.py connection format
"""

import toml
import urllib.parse
from sqlalchemy import create_engine, text

def test_updated_connection():
    try:
        # Load secrets
        secrets = toml.load('.streamlit/secrets.toml')
        db_config = secrets["db"]

        # Build connection string using available SQL Server driver (same as updated db.py)
        connection_string = (
            f"DRIVER={{SQL Server}};"
            f"SERVER={db_config['server']};"
            f"DATABASE={db_config['database']};"
            f"UID={db_config['username']};"
            f"PWD={db_config['password']};"
        )

        # Create SQLAlchemy URL (same as updated db.py)
        connection_url = f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(connection_string)}"

        print("=== TESTING UPDATED CONNECTION FORMAT ===")
        print(f"Connection string format: DRIVER={{SQL Server}};SERVER=...;DATABASE=...;UID=...;PWD=...")
        print(f"SQLAlchemy URL (first 50 chars): {connection_url[:50]}...")

        # Create engine
        engine = create_engine(connection_url, echo=False)
        print("✓ Engine created successfully")

        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            test_value = result.fetchone()[0]
            print(f"✓ Connection successful! Test query returned: {test_value}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")

        # Provide specific guidance based on error type
        if "40615" in str(e):
            print("\n🔥 FIREWALL ISSUE DETECTED!")
            print("This error means your IP address is blocked by Azure SQL firewall.")
            print("Contact your Azure admin to whitelist your IP: 14.191.222.209")
        elif "IM002" in str(e):
            print("\n🔧 DRIVER ISSUE!")
            print("ODBC driver still not found - this shouldn't happen with 'SQL Server' driver")
        elif "28000" in str(e):
            print("\n🔑 AUTHENTICATION ISSUE!")
            print("Invalid username/password credentials")

if __name__ == "__main__":
    test_updated_connection()