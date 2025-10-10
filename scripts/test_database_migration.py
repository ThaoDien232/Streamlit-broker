"""
Test script to verify database migration is working correctly.
Compares database values with CSV values for validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.brokerage_data import get_calc_metric_value, load_brokerage_metrics
from utils.market_index_data import load_market_liquidity_data

print("=" * 80)
print("DATABASE MIGRATION TEST")
print("=" * 80)

# Test 1: Load brokerage data from database
print("\n1. Testing brokerage data load from database...")
try:
    df_db = load_brokerage_metrics(ticker='SSI', start_year=2024, end_year=2024)
    if not df_db.empty:
        print(f"✅ Loaded {len(df_db)} records for SSI 2024")
        print(f"   Quarters available: {df_db['QUARTER_LABEL'].unique()}")
    else:
        print("❌ No data loaded")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Compare specific values
print("\n2. Testing specific metric values (SSI 2024 Q1)...")

# Load CSV for comparison
print("   Loading CSV data...")
df_csv = pd.read_csv('sql/Combined_Financial_Data.csv')

test_metrics = [
    'NET_BROKERAGE_INCOME',
    'PBT',
    'NPAT',
    'TOTAL_ASSETS',
    'TOTAL_EQUITY',
    'ROE'
]

print("\n   Comparing database vs CSV values:")
print(f"   {'Metric':<30} {'Database':<20} {'CSV':<20} {'Match?':<10}")
print("   " + "-" * 80)

for metric in test_metrics:
    try:
        # Get from database
        db_value = get_calc_metric_value(df_db, 'SSI', 2024, 1, metric)

        # Get from CSV
        csv_result = df_csv[
            (df_csv['TICKER'] == 'SSI') &
            (df_csv['YEARREPORT'] == 2024) &
            (df_csv['LENGTHREPORT'] == 1) &
            (df_csv['METRIC_CODE'] == metric)
        ]

        if not csv_result.empty:
            csv_value = csv_result.iloc[0]['VALUE']
        else:
            csv_value = 0

        # Compare (allow small floating point differences)
        match = "✅" if abs(db_value - csv_value) < 1 else "❌"

        print(f"   {metric:<30} {db_value:>19,.0f} {csv_value:>19,.0f} {match:<10}")

    except Exception as e:
        print(f"   {metric:<30} ERROR: {e}")

# Test 3: Market liquidity data
print("\n3. Testing market liquidity data from database...")
try:
    liquidity_df = load_market_liquidity_data(start_year=2024)
    if not liquidity_df.empty:
        print(f"✅ Loaded market liquidity data")
        print(f"\n   Sample data:")
        print(liquidity_df.head().to_string(index=False))
    else:
        print("❌ No liquidity data loaded")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Check available tickers
print("\n4. Testing available tickers from database...")
try:
    from utils.brokerage_data import get_available_tickers
    tickers = get_available_tickers()
    print(f"✅ Found {len(tickers)} tickers: {', '.join(tickers[:10])}...")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Check available quarters
print("\n5. Testing available quarters from database...")
try:
    from utils.brokerage_data import get_available_quarters
    quarters = get_available_quarters('SSI')
    print(f"✅ Found {len(quarters)} quarters for SSI")
    print(f"   Latest 5: {', '.join(quarters[:5])}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("DATABASE MIGRATION TEST COMPLETE")
print("=" * 80)
