"""
Database exploration script to map KEYCODEs from BrokerageMetrics table.
This will help us build the mapping between our CSV format and database format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import run_query
import pandas as pd

def explore_keycodes():
    """Get all unique KEYCODEs from BrokerageMetrics table."""

    print("=" * 80)
    print("EXPLORING BROKERAGEMETRICS KEYCODES")
    print("=" * 80)

    # Query 1: Get all unique KEYCODEs with their names
    query1 = """
    SELECT DISTINCT
        KEYCODE,
        KEYCODE_NAME
    FROM dbo.BrokerageMetrics
    WHERE TICKER = 'SSI'
      AND YEARREPORT = 2024
      AND LENGTHREPORT = 1
    ORDER BY KEYCODE
    """

    print("\n1. Fetching all KEYCODEs for SSI 2024 Q1...")
    df_keycodes = run_query(query1)

    if not df_keycodes.empty:
        print(f"\n✅ Found {len(df_keycodes)} unique KEYCODEs")

        # Save to CSV for analysis
        output_file = 'sql/database_keycodes_mapping.csv'
        df_keycodes.to_csv(output_file, index=False)
        print(f"✅ Saved to {output_file}")

        # Display by category
        print("\n" + "=" * 80)
        print("CALCULATED METRICS (likely what we need):")
        print("=" * 80)
        calc_metrics = df_keycodes[
            ~df_keycodes['KEYCODE'].str.contains(r'^(BS\.|IS\.|7\.|8\.)', regex=True, na=False)
        ]
        for _, row in calc_metrics.iterrows():
            print(f"  {row['KEYCODE']:<40} → {row['KEYCODE_NAME']}")

        print("\n" + "=" * 80)
        print("BALANCE SHEET ITEMS (BS.*):")
        print("=" * 80)
        bs_items = df_keycodes[df_keycodes['KEYCODE'].str.contains(r'^BS\.', regex=True, na=False)]
        print(f"  Found {len(bs_items)} Balance Sheet items (BS.1 - BS.169)")
        print("\n  Looking for TOTAL items:")
        total_items = bs_items[bs_items['KEYCODE_NAME'].str.contains('TOTAL|Total|total', regex=True, na=False)]
        for _, row in total_items.iterrows():
            print(f"    {row['KEYCODE']:<20} → {row['KEYCODE_NAME']}")

        print("\n  Looking for EQUITY/CAPITAL items:")
        equity_items = bs_items[bs_items['KEYCODE_NAME'].str.contains('EQUITY|Equity|equity|CAPITAL|Capital|capital|SHAREHOLDER', regex=True, na=False)]
        for _, row in equity_items.iterrows():
            print(f"    {row['KEYCODE']:<20} → {row['KEYCODE_NAME']}")

        print("\n  Looking for ASSET items:")
        asset_items = bs_items[bs_items['KEYCODE_NAME'].str.contains('ASSET|Asset|asset', regex=True, na=False)][:10]  # First 10
        for _, row in asset_items.iterrows():
            print(f"    {row['KEYCODE']:<20} → {row['KEYCODE_NAME']}")

        print("\n" + "=" * 80)
        print("INVESTMENT PORTFOLIO (7.* - Cost, 8.* - Market Value):")
        print("=" * 80)
        portfolio_items = df_keycodes[
            df_keycodes['KEYCODE'].str.contains(r'^(7\.|8\.)', regex=True, na=False)
        ]
        print(f"  Found {len(portfolio_items)} investment portfolio items")
        for _, row in portfolio_items.head(20).iterrows():  # First 20
            print(f"    {row['KEYCODE']:<40} → {row['KEYCODE_NAME']}")

    else:
        print("❌ No KEYCODEs found. Check database connection.")

    # Query 2: Check if we have data from 2017 onwards
    query2 = """
    SELECT
        YEARREPORT,
        COUNT(DISTINCT TICKER) as broker_count,
        COUNT(DISTINCT KEYCODE) as keycode_count,
        COUNT(*) as total_records
    FROM dbo.BrokerageMetrics
    WHERE YEARREPORT >= 2017
      AND LENGTHREPORT BETWEEN 1 AND 4
    GROUP BY YEARREPORT
    ORDER BY YEARREPORT DESC
    """

    print("\n" + "=" * 80)
    print("DATA AVAILABILITY (2017 onwards, quarterly only):")
    print("=" * 80)
    df_years = run_query(query2)

    if not df_years.empty:
        print(f"\n{'Year':<10}{'Brokers':<15}{'KEYCODEs':<15}{'Total Records':<20}")
        print("-" * 60)
        for _, row in df_years.iterrows():
            print(f"{row['YEARREPORT']:<10}{row['broker_count']:<15}{row['keycode_count']:<15}{row['total_records']:<20}")

    # Query 3: Find ROE, ROA in database
    query3 = """
    SELECT DISTINCT
        KEYCODE,
        KEYCODE_NAME
    FROM dbo.BrokerageMetrics
    WHERE KEYCODE_NAME LIKE '%ROE%' OR KEYCODE_NAME LIKE '%ROA%'
       OR KEYCODE LIKE '%ROE%' OR KEYCODE LIKE '%ROA%'
    """

    print("\n" + "=" * 80)
    print("SEARCHING FOR ROE/ROA:")
    print("=" * 80)
    df_roe_roa = run_query(query3)

    if not df_roe_roa.empty:
        for _, row in df_roe_roa.iterrows():
            print(f"  {row['KEYCODE']:<40} → {row['KEYCODE_NAME']}")
    else:
        print("  ❌ ROE/ROA not found in database - need to calculate")

    # Query 4: Sample data values to verify
    query4 = """
    SELECT TOP 5
        TICKER,
        YEARREPORT,
        LENGTHREPORT,
        QUARTER_LABEL,
        KEYCODE,
        VALUE
    FROM dbo.BrokerageMetrics
    WHERE TICKER = 'SSI'
      AND YEARREPORT = 2024
      AND LENGTHREPORT = 1
      AND KEYCODE IN ('PBT', 'NPAT', 'Net_Brokerage_Income')
    """

    print("\n" + "=" * 80)
    print("SAMPLE DATA (SSI 2024 Q1):")
    print("=" * 80)
    df_sample = run_query(query4)

    if not df_sample.empty:
        print(df_sample.to_string(index=False))

if __name__ == "__main__":
    explore_keycodes()
