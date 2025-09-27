#!/usr/bin/env python3
"""
Data Completeness Check Script
Compare Combined_Financial_Data.csv with original IS_security.csv and BS_security.csv
to identify missing data points.
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def load_data():
    """Load all three datasets"""
    print("Loading datasets...")
    
    # Load IS data
    is_data = pd.read_csv('sql/IS_security.csv')
    print(f"IS_security.csv: {len(is_data)} records")
    
    # Load BS data
    bs_data = pd.read_csv('sql/BS_security.csv')
    print(f"BS_security.csv: {len(bs_data)} records")
    
    # Load combined data
    combined_data = pd.read_csv('sql/Combined_Financial_Data.csv')
    print(f"Combined_Financial_Data.csv: {len(combined_data)} records")
    
    return is_data, bs_data, combined_data

def get_broker_data(data, ticker_col='TICKER'):
    """Get data grouped by broker (ticker)"""
    return data.groupby(ticker_col)

def analyze_is_completeness(is_data, combined_data):
    """Analyze Income Statement data completeness"""
    print("\n=== INCOME STATEMENT DATA COMPLETENESS ANALYSIS ===")
    
    # Get unique tickers from both datasets, filtering out NaN values
    is_tickers = set(is_data['TICKER'].dropna().unique())
    combined_tickers = set(combined_data['TICKER'].dropna().unique())
    
    print(f"IS tickers: {len(is_tickers)} - {sorted(is_tickers)}")
    print(f"Combined tickers: {len(combined_tickers)} - {sorted(combined_tickers)}")
    
    missing_data = defaultdict(lambda: defaultdict(list))
    
    # Check for missing data for each broker
    for ticker in is_tickers:
        if ticker not in combined_tickers:
            print(f"\nWARNING: {ticker} completely missing from combined data!")
            continue
            
        # Get data for this ticker
        is_ticker_data = is_data[is_data['TICKER'] == ticker]
        combined_ticker_data = combined_data[combined_data['TICKER'] == ticker]
        
        print(f"\n--- Checking {ticker} ---")
        print(f"IS records: {len(is_ticker_data)}")
        print(f"Combined records: {len(combined_ticker_data)}")
        
        # Check for missing quarters/years
        is_periods = set(zip(is_ticker_data['YEARREPORT'], is_ticker_data['LENGTHREPORT']))
        combined_periods = set(zip(combined_ticker_data['YEARREPORT'], combined_ticker_data['LENGTHREPORT']))
        
        missing_periods = is_periods - combined_periods
        if missing_periods:
            print(f"Missing periods for {ticker}: {missing_periods}")
            missing_data[ticker]['missing_periods'] = list(missing_periods)
        
        # Check for specific IS metrics (focusing on the ones mentioned)
        is_columns = [col for col in is_data.columns if col.startswith('ISS') or col.startswith('ISA')]
        
        for year, length in is_periods.intersection(combined_periods):
            is_record = is_ticker_data[(is_ticker_data['YEARREPORT'] == year) & 
                                     (is_ticker_data['LENGTHREPORT'] == length)]
            combined_record = combined_ticker_data[(combined_ticker_data['YEARREPORT'] == year) & 
                                                 (combined_ticker_data['LENGTHREPORT'] == length)]
            
            if len(is_record) > 0 and len(combined_record) > 0:
                is_record = is_record.iloc[0]
                
                # Check for missing IS.9 (Income from derivatives) and IS.31 (Loss from derivatives)
                # These correspond to specific column names in the IS data
                for col in is_columns:
                    if pd.notna(is_record[col]) and is_record[col] != 0:
                        # Check if this data exists in combined data
                        matching_combined = combined_record[
                            (combined_record['METRIC_CODE'] == get_metric_code_from_column(col)) |
                            (combined_record['KEYCODE'].str.contains(get_is_keycode_from_column(col), na=False))
                        ] if not combined_record.empty else pd.DataFrame()
                        
                        if matching_combined.empty:
                            missing_data[ticker]['missing_metrics'].append({
                                'period': f"{year}-{length}",
                                'column': col,
                                'value': is_record[col],
                                'estimated_keycode': get_is_keycode_from_column(col)
                            })
    
    return missing_data

def get_metric_code_from_column(column):
    """Map IS column names to metric codes"""
    # This is a simplified mapping - you might need to adjust based on your data structure
    mapping = {
        'ISA9': 'IS.9',   # Income from derivatives
        'ISA31': 'IS.31', # Loss from derivatives
        # Add more mappings as needed
    }
    return mapping.get(column, column)

def get_is_keycode_from_column(column):
    """Map IS column names to IS keycodes"""
    # Simplified mapping for IS keycodes
    mapping = {
        'ISA9': 'IS.9',
        'ISA31': 'IS.31',
        # Add more mappings as needed
    }
    return mapping.get(column, 'IS.0')

def analyze_bs_completeness(bs_data, combined_data):
    """Analyze Balance Sheet data completeness"""
    print("\n=== BALANCE SHEET DATA COMPLETENESS ANALYSIS ===")
    
    # Similar analysis for BS data
    bs_tickers = set(bs_data['TICKER'].dropna().unique())
    combined_tickers = set(combined_data['TICKER'].dropna().unique())
    
    print(f"BS tickers: {len(bs_tickers)} - {sorted(bs_tickers)}")
    
    missing_data = defaultdict(lambda: defaultdict(list))
    
    for ticker in bs_tickers:
        if ticker not in combined_tickers:
            print(f"\nWARNING: {ticker} completely missing from combined data!")
            continue
            
        bs_ticker_data = bs_data[bs_data['TICKER'] == ticker]
        combined_ticker_data = combined_data[combined_data['TICKER'] == ticker]
        
        print(f"\n--- Checking {ticker} ---")
        print(f"BS records: {len(bs_ticker_data)}")
        print(f"Combined records with BS data: {len(combined_ticker_data[combined_ticker_data['STATEMENT_TYPE'] == 'BS'])}")
    
    return missing_data

def generate_missing_data_report(is_missing, bs_missing, combined_data):
    """Generate comprehensive missing data report"""
    print("\n" + "="*60)
    print("COMPREHENSIVE MISSING DATA REPORT")
    print("="*60)
    
    # Get the specific metrics mentioned by the user
    print("\n1. SPECIFIC MISSING METRICS CHECK:")
    print("-" * 40)
    
    # Check for IS.9 and IS.31 for SSI specifically
    ssi_data = combined_data[combined_data['TICKER'] == 'SSI']
    
    print(f"SSI total records: {len(ssi_data)}")
    
    is9_records = ssi_data[ssi_data['KEYCODE'] == 'IS.9']
    is31_records = ssi_data[ssi_data['KEYCODE'] == 'IS.31']
    
    print(f"SSI IS.9 (Income from derivatives) records: {len(is9_records)}")
    print(f"SSI IS.31 (Loss from derivatives) records: {len(is31_records)}")
    
    if len(is9_records) == 0:
        print("✗ CONFIRMED: IS.9 (Income from derivatives) missing for SSI")
    else:
        print("✓ IS.9 (Income from derivatives) found for SSI")
        
    if len(is31_records) == 0:
        print("✗ CONFIRMED: IS.31 (Loss from derivatives) missing for SSI")
    else:
        print("✓ IS.31 (Loss from derivatives) found for SSI")
    
    # Check all brokers for these specific metrics
    print("\n2. IS.9 and IS.31 AVAILABILITY BY BROKER:")
    print("-" * 40)
    
    all_tickers = sorted(combined_data['TICKER'].dropna().unique())
    for ticker in all_tickers:
        ticker_data = combined_data[combined_data['TICKER'] == ticker]
        is9_count = len(ticker_data[ticker_data['KEYCODE'] == 'IS.9'])
        is31_count = len(ticker_data[ticker_data['KEYCODE'] == 'IS.31'])
        
        status_is9 = "✓" if is9_count > 0 else "✗"
        status_is31 = "✓" if is31_count > 0 else "✗"
        
        print(f"{ticker}: IS.9 {status_is9} ({is9_count} records) | IS.31 {status_is31} ({is31_count} records)")
    
    # Check for other common missing metrics
    print("\n3. OTHER POTENTIALLY MISSING METRICS:")
    print("-" * 40)
    
    # Get all unique keycodes from combined data
    all_keycodes = set(combined_data['KEYCODE'].unique())
    
    # Common IS metrics that might be missing
    common_is_metrics = [
        'IS.3', 'IS.4', 'IS.5', 'IS.8', 'IS.9', 'IS.27', 'IS.24', 
        'IS.25', 'IS.26', 'IS.28', 'IS.29', 'IS.31', 'IS.32'
    ]
    
    missing_keycodes = []
    for keycode in common_is_metrics:
        if keycode not in all_keycodes:
            missing_keycodes.append(keycode)
        else:
            # Check how many brokers have this metric
            brokers_with_metric = combined_data[combined_data['KEYCODE'] == keycode]['TICKER'].nunique()
            total_brokers = combined_data['TICKER'].nunique()
            print(f"{keycode}: Available for {brokers_with_metric}/{total_brokers} brokers")
    
    if missing_keycodes:
        print(f"\nCompletely missing keycodes: {missing_keycodes}")
    
    return {
        'missing_keycodes': missing_keycodes,
        'is_missing': is_missing,
        'bs_missing': bs_missing
    }

def main():
    """Main execution function"""
    try:
        # Load data
        is_data, bs_data, combined_data = load_data()
        
        # Analyze completeness
        is_missing = analyze_is_completeness(is_data, combined_data)
        bs_missing = analyze_bs_completeness(bs_data, combined_data)
        
        # Generate comprehensive report
        report = generate_missing_data_report(is_missing, bs_missing, combined_data)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return report
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()