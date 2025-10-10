"""
Quick database check - get KEYCODEs we need for migration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import run_query
import pandas as pd

def quick_check():
    """Get essential KEYCODEs quickly."""

    print("Checking database KEYCODEs...")

    # Just get calculated metrics and totals
    query = """
    SELECT TOP 500
        KEYCODE,
        KEYCODE_NAME
    FROM dbo.BrokerageMetrics
    WHERE TICKER = 'SSI'
      AND YEARREPORT = 2024
      AND LENGTHREPORT = 1
      AND (
          KEYCODE NOT LIKE 'BS.%'
          OR KEYCODE_NAME LIKE '%TOTAL%'
          OR KEYCODE_NAME LIKE '%EQUITY%'
          OR KEYCODE_NAME LIKE '%CAPITAL%'
      )
    ORDER BY KEYCODE
    """

    df = run_query(query)

    if not df.empty:
        # Remove duplicates
        df = df.drop_duplicates(subset=['KEYCODE'])

        print(f"\n✅ Found {len(df)} KEYCODEs\n")

        # Group by type
        calc_metrics = df[~df['KEYCODE'].str.contains(r'^(BS\.|IS\.|7\.|8\.)', regex=True, na=False)]
        bs_totals = df[df['KEYCODE'].str.contains(r'^BS\.', regex=True, na=False)]

        print("CALCULATED METRICS:")
        print("-" * 80)
        for _, row in calc_metrics.iterrows():
            print(f"  {row['KEYCODE']:<40} | {row['KEYCODE_NAME']}")

        print("\n\nBALANCE SHEET TOTALS/EQUITY:")
        print("-" * 80)
        for _, row in bs_totals.iterrows():
            print(f"  {row['KEYCODE']:<40} | {row['KEYCODE_NAME']}")

        # Save for reference
        df.to_csv('sql/db_keycodes_quick.csv', index=False)
        print(f"\n✅ Saved to sql/db_keycodes_quick.csv")
    else:
        print("❌ No data returned")

if __name__ == "__main__":
    quick_check()
