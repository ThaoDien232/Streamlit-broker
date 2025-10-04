#!/usr/bin/env python3
"""Test database connection without Streamlit context"""

import toml
import pyodbc
import pymssql
from sqlalchemy import create_engine, text
import urllib.parse

def test_connection_methods():
    """Test all connection methods using secrets.toml directly"""

    try:
        # Load secrets directly from file
        secrets = toml.load('.streamlit/secrets.toml')
        db_config = secrets['db']

        print("=== TESTING ALL CONNECTION METHODS ===")
        print(f"Server: {db_config['server']}")
        print(f"Database: {db_config['database']}")
        print(f"Username: {db_config['username']}")

        success = False

        # Method 1: pyodbc with SQL Server driver
        print("\n--- Method 1: pyodbc + SQL Server driver ---")
        try:
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

            conn = pyodbc.connect(connection_string, timeout=30)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"✓ SUCCESS: pyodbc connection works! Result: {result[0]}")
            cursor.close()
            conn.close()
            success = True

        except Exception as e:
            print(f"✗ FAILED: {e}")

        # Method 2: pymssql
        if not success:
            print("\n--- Method 2: pymssql ---")
            try:
                conn = pymssql.connect(
                    server=db_config['server'],
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database'],
                    port=int(db_config['port']),
                    charset='UTF-8',
                    timeout=30,
                    login_timeout=30
                )

                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                print(f"✓ SUCCESS: pymssql connection works! Result: {result[0]}")
                cursor.close()
                conn.close()
                success = True

            except Exception as e:
                print(f"✗ FAILED: {e}")

        # Method 3: SQLAlchemy
        if not success:
            print("\n--- Method 3: SQLAlchemy ---")
            try:
                connection_url = db_config['url']
                # Fix driver name
                fixed_url = connection_url.replace("ODBC+Driver+18+for+SQL+Server", "SQL+Server")

                engine = create_engine(fixed_url, pool_pre_ping=True)
                conn = engine.connect()
                result = conn.execute(text("SELECT 1"))
                row = result.fetchone()
                print(f"✓ SUCCESS: SQLAlchemy connection works! Result: {row[0]}")
                conn.close()
                success = True

            except Exception as e:
                print(f"✗ FAILED: {e}")

        if success:
            print(f"\n🎉 At least one connection method worked!")
        else:
            print(f"\n❌ All connection methods failed")

    except Exception as e:
        print(f"❌ Error loading secrets: {e}")

if __name__ == "__main__":
    test_connection_methods()