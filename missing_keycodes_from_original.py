#!/usr/bin/env python3
"""
Find keycodes that exist in original files but are completely missing from combined data
"""

import pandas as pd
import numpy as np

def find_completely_missing_keycodes():
    print("Loading datasets...")
    
    # Load combined data to get available keycodes
    combined_data = pd.read_csv('sql/Combined_Financial_Data.csv')
    combined_keycodes = set(combined_data['KEYCODE'].dropna().unique())
    
    print(f"Combined data keycodes: {len(combined_keycodes)}")
    
    # Load original data sample to check column structure
    is_data = pd.read_csv('sql/IS_security.csv', nrows=1000)  # Sample first 1000 rows
    bs_data = pd.read_csv('sql/BS_security.csv', nrows=1000)  # Sample first 1000 rows
    
    print(f"IS sample data: {len(is_data)} records")
    print(f"BS sample data: {len(bs_data)} records")
    
    # Get IS column names that could map to keycodes
    is_columns = [col for col in is_data.columns if col.startswith(('ISS', 'ISA'))]
    bs_columns = [col for col in bs_data.columns if col.startswith('BSS') or col.startswith('BSA')]
    
    print(f"\nIS columns in original data: {len(is_columns)}")
    print(f"BS columns in original data: {len(bs_columns)}")
    
    # Load the data.py file to see the keycode mapping
    try:
        with open('utils/data.py', 'r') as f:
            data_py_content = f.read()
    except:
        print("Could not read utils/data.py")
        data_py_content = ""
    
    # Check if there are any keycode mappings we can extract
    print(f"\n" + "="*60)
    print("ANALYZING POTENTIAL MISSING KEYCODES")
    print("="*60)
    
    # Common IS keycodes that might be missing
    expected_is_keycodes = []
    for i in range(1, 100):
        expected_is_keycodes.append(f"IS.{i}")
    
    expected_bs_keycodes = []
    for i in range(1, 200):
        expected_bs_keycodes.append(f"BS.{i}")
    
    # Find completely missing IS keycodes
    missing_is = [k for k in expected_is_keycodes if k not in combined_keycodes]
    missing_bs = [k for k in expected_bs_keycodes if k not in combined_keycodes]
    
    print(f"COMPLETELY MISSING IS KEYCODES:")
    print("-" * 40)
    for i, keycode in enumerate(missing_is[:30]):  # Show first 30
        print(f"{keycode:6s}", end="  ")
        if (i + 1) % 8 == 0:
            print()
    print(f"\n... and {len(missing_is) - 30} more" if len(missing_is) > 30 else "")
    
    print(f"\nCOMPLETELY MISSING BS KEYCODES:")
    print("-" * 40)
    for i, keycode in enumerate(missing_bs[:40]):  # Show first 40
        print(f"{keycode:6s}", end="  ")
        if (i + 1) % 8 == 0:
            print()
    print(f"\n... and {len(missing_bs) - 40} more" if len(missing_bs) > 40 else "")
    
    # Check specific keycodes from your original list
    your_list_keycodes = ['IS.3','IS.4','IS.5','IS.8','IS.9','IS.27','IS.24','IS.25','IS.26','IS.28','IS.29','IS.31','IS.32']
    
    print(f"\n" + "="*60)
    print("STATUS OF YOUR SPECIFIC KEYCODES")
    print("="*60)
    
    for keycode in your_list_keycodes:
        if keycode in combined_keycodes:
            broker_count = combined_data[combined_data['KEYCODE'] == keycode]['TICKER'].nunique()
            record_count = len(combined_data[combined_data['KEYCODE'] == keycode])
            print(f"{keycode:6s}: PRESENT - {broker_count:3d} brokers, {record_count:4d} records")
        else:
            print(f"{keycode:6s}: COMPLETELY MISSING")
    
    # Check for any patterns in IS column names from original data
    print(f"\n" + "="*60)
    print("SAMPLE IS COLUMNS FROM ORIGINAL DATA")
    print("="*60)
    print("First 20 IS columns:")
    for col in is_columns[:20]:
        print(f"  {col}")
    print(f"... and {len(is_columns) - 20} more" if len(is_columns) > 20 else "")
    
    print(f"\n" + "="*60)
    print("SAMPLE BS COLUMNS FROM ORIGINAL DATA")
    print("="*60)
    print("First 20 BS columns:")
    for col in bs_columns[:20]:
        print(f"  {col}")
    print(f"... and {len(bs_columns) - 20} more" if len(bs_columns) > 20 else "")
    
    # Check if we can find any data in these columns
    print(f"\n" + "="*60)
    print("CHECKING FOR DATA IN ORIGINAL IS COLUMNS")
    print("="*60)
    
    non_null_columns = []
    for col in is_columns[:50]:  # Check first 50 columns
        non_null_count = is_data[col].notna().sum()
        non_zero_count = (is_data[col] != 0).sum()
        if non_null_count > 0 and non_zero_count > 0:
            non_null_columns.append((col, non_null_count, non_zero_count))
    
    print("IS columns with data (non-null and non-zero):")
    for col, null_count, zero_count in non_null_columns:
        print(f"  {col}: {null_count} non-null, {zero_count} non-zero values")
    
    return {
        'missing_is': missing_is,
        'missing_bs': missing_bs,
        'your_keycodes_status': [(k, k in combined_keycodes) for k in your_list_keycodes],
        'original_is_columns': is_columns,
        'original_bs_columns': bs_columns
    }

if __name__ == "__main__":
    try:
        result = find_completely_missing_keycodes()
        print(f"\n" + "="*60)
        print("MISSING KEYCODE ANALYSIS COMPLETE")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()