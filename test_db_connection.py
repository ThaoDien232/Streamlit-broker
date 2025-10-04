#!/usr/bin/env python3
"""
Test script to explore the database schema and understand available data.
Run this to see what tables and data are available for valuation analysis.
"""

import sys
sys.path.append('.')

from utils.db import test_connection, get_available_tables, get_table_info, run_query
import pandas as pd

def main():
    print("Testing Database Connection...")
    print("=" * 50)

    # Test basic connection
    if test_connection():
        print("SUCCESS: Database connection successful!")
    else:
        print("ERROR: Database connection failed!")
        return

    print("\n" + "=" * 50)
    print("Available Tables:")
    print("=" * 50)

    # Get list of tables
    try:
        tables_df = get_available_tables()
        if not tables_df.empty:
            for _, row in tables_df.head(20).iterrows():
                print(f"  {row['TABLE_SCHEMA']}.{row['TABLE_NAME']}")
            if len(tables_df) > 20:
                print(f"  ... and {len(tables_df) - 20} more tables")
        else:
            print("No tables found or insufficient permissions")
    except Exception as e:
        print(f"Error getting tables: {e}")

    print("\n" + "=" * 50)
    print("Looking for Market/Valuation Data:")
    print("=" * 50)

    # Search for tables that might contain market/valuation data
    search_terms = ['market', 'valuation', 'price', 'stock', 'ticker', 'equity', 'pe', 'pb']

    try:
        tables_df = get_available_tables()
        if not tables_df.empty:
            for term in search_terms:
                matching_tables = tables_df[
                    tables_df['TABLE_NAME'].str.contains(term, case=False, na=False)
                ]['TABLE_NAME'].unique()

                if len(matching_tables) > 0:
                    print(f"\nTables containing '{term}':")
                    for table in matching_tables:
                        print(f"  {table}")
    except Exception as e:
        print(f"Error searching tables: {e}")

    print("\n" + "=" * 50)
    print("Sample Data from Key Tables:")
    print("=" * 50)

    # Try to get sample data from likely tables
    potential_tables = ['Market_Data', 'Stock_Data', 'Valuation_Data', 'Price_Data', 'Financial_Data']

    for table_name in potential_tables:
        try:
            print(f"\nTable: {table_name}")
            print("-" * 30)

            # Get table structure
            columns_df = get_table_info(table_name)
            if not columns_df.empty:
                print("Columns:")
                for _, col in columns_df.head(10).iterrows():
                    print(f"  {col['COLUMN_NAME']} ({col['DATA_TYPE']})")

                # Get sample data
                sample_query = f"SELECT TOP 3 * FROM dbo.{table_name}"
                sample_df = run_query(sample_query)

                if not sample_df.empty:
                    print(f"\nSample data (first 3 rows):")
                    print(sample_df.to_string())
                else:
                    print("No sample data available")
            else:
                print(f"Table {table_name} not found or no access")

        except Exception as e:
            print(f"Error accessing {table_name}: {e}")

    print("\n" + "=" * 50)
    print("Checking for Broker/Stock Tickers:")
    print("=" * 50)

    # Try to find broker tickers in the data
    broker_tickers = ['SSI', 'VCI', 'HCM', 'VIX', 'VND', 'MBS', 'TCBS', 'FPTS']

    try:
        # Look for these tickers in Market_Data table if it exists
        for ticker in broker_tickers[:3]:  # Test first 3
            query = f"""
            SELECT TOP 5 *
            FROM dbo.Market_Data
            WHERE TICKER = '{ticker}'
            ORDER BY TRADE_DATE DESC
            """
            result = run_query(query)

            if not result.empty:
                print(f"\nFound data for {ticker}:")
                print(result.to_string())
                break
        else:
            print("No broker ticker data found in Market_Data table")

    except Exception as e:
        print(f"Error checking broker data: {e}")

if __name__ == "__main__":
    main()