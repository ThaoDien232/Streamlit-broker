"""
Test database connection with minimal query.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import run_query, test_connection
import pandas as pd

print("Testing database connection...")

# Test basic connection
if test_connection():
    print("✅ Database connection successful\n")

    # Very simple query - just get 10 records
    query = """
    SELECT TOP 10
        KEYCODE,
        KEYCODE_NAME
    FROM dbo.BrokerageMetrics
    WHERE TICKER = 'SSI'
    """

    print("Running simple query...")
    df = run_query(query)

    if not df.empty:
        print(f"\n✅ Query successful! Got {len(df)} records:\n")
        print(df.to_string(index=False))
    else:
        print("❌ Query returned no data")
else:
    print("❌ Database connection failed")
