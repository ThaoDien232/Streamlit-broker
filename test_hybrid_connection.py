#!/usr/bin/env python3
"""Test the hybrid connection approach for Azure SQL"""

import toml
import sys
sys.path.append('.')

from utils.db import test_connection, get_connection

def test_hybrid_approach():
    print("=== TESTING HYBRID CONNECTION APPROACH ===")

    try:
        # Test the connection
        success = test_connection()
        print(f"Connection test result: {'SUCCESS' if success else 'FAILED'}")

        if success:
            # Test actual connection to see which method worked
            try:
                with get_connection() as conn:
                    connection_type = type(conn).__name__
                    print(f"Connection established using: {connection_type}")

                    # Test a simple query
                    if hasattr(conn, 'cursor'):
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1 as test_value")
                        result = cursor.fetchone()
                        print(f"Test query result: {result[0]}")
                        cursor.close()
                    else:
                        # SQLAlchemy connection
                        from sqlalchemy import text
                        result = conn.execute(text("SELECT 1 as test_value"))
                        row = result.fetchone()
                        print(f"Test query result: {row[0]}")

                print("SUCCESS: Hybrid connection approach working!")

            except Exception as query_error:
                print(f"Connection succeeded but query failed: {query_error}")

    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_hybrid_approach()