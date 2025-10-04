#!/usr/bin/env python3
"""Test the updated pymssql connection"""

import toml
import pymssql

def test_pymssql_connection():
    try:
        # Load secrets
        secrets = toml.load('.streamlit/secrets.toml')
        db_config = secrets["db"]

        print("=== TESTING PYMSSQL CONNECTION ===")
        print(f"Server: {db_config['server']}")
        print(f"Database: {db_config['database']}")
        print(f"Port: {db_config.get('port', 1433)}")

        # Create direct pymssql connection (no ODBC drivers needed)
        connection = pymssql.connect(
            server=db_config['server'],
            user=db_config['username'],
            password=db_config['password'],
            database=db_config['database'],
            port=int(db_config.get('port', 1433)),
            charset='UTF-8',
            autocommit=False
        )

        print("SUCCESS: Connection established!")

        # Test query
        cursor = connection.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()
        print(f"SUCCESS: Test query returned: {result[0]}")

        cursor.close()
        connection.close()

    except Exception as e:
        print(f"FAILED: {e}")

        # Provide specific guidance
        if "40615" in str(e):
            print("FIREWALL ISSUE: Contact Azure admin to whitelist your IP")
        elif "18456" in str(e):
            print("AUTHENTICATION ISSUE: Check username/password")
        elif "timeout" in str(e).lower():
            print("TIMEOUT ISSUE: Check server address and port")

if __name__ == "__main__":
    test_pymssql_connection()