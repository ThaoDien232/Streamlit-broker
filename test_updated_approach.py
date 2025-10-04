#!/usr/bin/env python3
"""Test the updated approach that matches reference script"""

import toml
import pymssql

def test_updated_approach():
    try:
        # Load secrets
        secrets = toml.load('.streamlit/secrets.toml')
        db_config = secrets["db"]

        print("=== TESTING UPDATED APPROACH ===")
        print(f"Available keys in secrets: {list(db_config.keys())}")

        # Method 1: Try using the 'url' from secrets (SQLAlchemy format)
        if 'url' in db_config:
            print(f"Found URL in secrets: {db_config['url'][:50]}...")

            # Parse it properly for pymssql
            if db_config['url'].startswith('mssql+pyodbc://'):
                # Extract components from the URL format
                print("Detected SQLAlchemy URL format")

                # Use individual components instead
                server = db_config['server']
                if ',' not in server and 'port' in db_config:
                    server = f"{server},{db_config['port']}"

                print(f"Connecting to server: {server}")
                print(f"Database: {db_config['database']}")

                connection = pymssql.connect(
                    server=server,
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database'],
                    charset='UTF-8',
                    autocommit=False,
                    timeout=30,
                    login_timeout=30
                )

                print("SUCCESS: Connection established with updated approach!")

                cursor = connection.cursor()
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                print(f"SUCCESS: Test query returned: {result[0]}")

                cursor.close()
                connection.close()
            else:
                print("URL format not recognized")

    except Exception as e:
        print(f"FAILED: {e}")

        # Check specific error types
        if "40615" in str(e):
            print("STILL FIREWALL ISSUE: IP needs to be whitelisted")
        elif "18456" in str(e):
            print("AUTHENTICATION ISSUE: Check credentials")
        else:
            print(f"OTHER ISSUE: {type(e).__name__}")

if __name__ == "__main__":
    test_updated_approach()