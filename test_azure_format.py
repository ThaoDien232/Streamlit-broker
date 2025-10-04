#!/usr/bin/env python3
"""Test Azure SQL specific connection format"""

import toml
import pymssql

def test_azure_format():
    try:
        # Load secrets
        secrets = toml.load('.streamlit/secrets.toml')
        db_config = secrets["db"]

        print("=== TESTING AZURE SQL FORMAT ===")

        # For Azure SQL, try different server formats
        test_formats = [
            # Format 1: Just the server name
            db_config['server'],
            # Format 2: Server with port as separate parameter (don't include in server string)
            db_config['server'],
        ]

        for i, server_format in enumerate(test_formats, 1):
            try:
                print(f"\nTest {i}: Server='{server_format}', Port={db_config.get('port', 1433)}")

                connection = pymssql.connect(
                    server=server_format,
                    user=db_config['username'],
                    password=db_config['password'],
                    database=db_config['database'],
                    port=int(db_config.get('port', 1433)),
                    charset='UTF-8',
                    autocommit=False,
                    timeout=30,
                    login_timeout=30
                )

                print(f"SUCCESS with format {i}!")

                cursor = connection.cursor()
                cursor.execute("SELECT 1 as test")
                result = cursor.fetchone()
                print(f"Test query returned: {result[0]}")

                cursor.close()
                connection.close()
                return  # Exit on first success

            except Exception as e:
                print(f"Format {i} failed: {e}")
                error_code = str(e).split(',')[0].strip('(')
                print(f"Error code: {error_code}")

                if "40615" in str(e):
                    print("  -> FIREWALL issue")
                elif "20009" in str(e):
                    print("  -> Server/port connection issue")
                elif "18456" in str(e):
                    print("  -> Authentication issue")

        print("\nAll formats failed")

    except Exception as e:
        print(f"General error: {e}")

if __name__ == "__main__":
    test_azure_format()