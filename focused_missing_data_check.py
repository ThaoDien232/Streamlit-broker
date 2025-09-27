#!/usr/bin/env python3
"""
Focused Missing Data Check
Quick analysis of missing IS.9 and IS.31 data and other key metrics
"""

import pandas as pd
import numpy as np

def quick_analysis():
    print("Loading Combined Financial Data...")
    
    # Load only the combined data for quick analysis
    combined_data = pd.read_csv('sql/Combined_Financial_Data.csv')
    print(f"Combined data: {len(combined_data)} records")
    
    # Get all unique tickers
    all_tickers = sorted(combined_data['TICKER'].dropna().unique())
    print(f"\nTotal brokers in combined data: {len(all_tickers)}")
    
    # Focus on the specific missing metrics mentioned
    print("\n" + "="*60)
    print("FOCUSED MISSING DATA ANALYSIS")
    print("="*60)
    
    # 1. Check IS.9 and IS.31 for all brokers
    print("\n1. IS.9 (Income from derivatives) and IS.31 (Loss from derivatives):")
    print("-" * 60)
    
    missing_is9 = []
    missing_is31 = []
    
    for ticker in all_tickers:
        if ticker == 'Sector':  # Skip non-broker entries
            continue
            
        ticker_data = combined_data[combined_data['TICKER'] == ticker]
        is9_count = len(ticker_data[ticker_data['KEYCODE'] == 'IS.9'])
        is31_count = len(ticker_data[ticker_data['KEYCODE'] == 'IS.31'])
        
        status_is9 = "Y" if is9_count > 0 else "N"
        status_is31 = "Y" if is31_count > 0 else "N"
        
        print(f"{ticker:8s}: IS.9 {status_is9} ({is9_count:2d}) | IS.31 {status_is31} ({is31_count:2d})")
        
        if is9_count == 0:
            missing_is9.append(ticker)
        if is31_count == 0:
            missing_is31.append(ticker)
    
    # 2. Summary of missing data
    print(f"\n2. SUMMARY:")
    print("-" * 30)
    print(f"Brokers missing IS.9:  {len(missing_is9)}/{len([t for t in all_tickers if t != 'Sector'])}")
    print(f"Brokers missing IS.31: {len(missing_is31)}/{len([t for t in all_tickers if t != 'Sector'])}")
    
    if missing_is9:
        print(f"\nBrokers missing IS.9 (Income from derivatives):")
        for ticker in missing_is9:
            print(f"  - {ticker}")
    
    if missing_is31:
        print(f"\nBrokers missing IS.31 (Loss from derivatives):")
        for ticker in missing_is31:
            print(f"  - {ticker}")
    
    # 3. Check other key metrics from the list you provided
    print(f"\n3. OTHER KEY METRICS CHECK:")
    print("-" * 40)
    
    key_metrics = ['IS.3', 'IS.4', 'IS.5', 'IS.8', 'IS.9', 'IS.27', 'IS.24', 
                   'IS.25', 'IS.26', 'IS.28', 'IS.29', 'IS.31', 'IS.32']
    
    metric_availability = {}
    for metric in key_metrics:
        brokers_with_metric = combined_data[combined_data['KEYCODE'] == metric]['TICKER'].nunique()
        total_brokers = len([t for t in all_tickers if t != 'Sector'])
        metric_availability[metric] = (brokers_with_metric, total_brokers)
        
        status = "Y" if brokers_with_metric > total_brokers * 0.5 else "P" if brokers_with_metric > 0 else "N"
        print(f"{metric:6s}: {status} {brokers_with_metric:3d}/{total_brokers:3d} brokers ({brokers_with_metric/total_brokers*100:.1f}%)")
    
    # 4. Check specific brokers mentioned (SSI, etc.)
    print(f"\n4. SPECIFIC BROKER ANALYSIS:")
    print("-" * 35)
    
    important_brokers = ['SSI', 'VCI', 'VIX', 'HCM', 'TCBS', 'MBS', 'VDS']
    
    for broker in important_brokers:
        if broker in all_tickers:
            print(f"\n{broker}:")
            broker_data = combined_data[combined_data['TICKER'] == broker]
            
            for metric in ['IS.9', 'IS.31']:
                count = len(broker_data[broker_data['KEYCODE'] == metric])
                status = "Y" if count > 0 else "N"
                print(f"  {metric}: {status} ({count} records)")
                
                if count > 0:
                    # Show some sample values
                    sample = broker_data[broker_data['KEYCODE'] == metric][['YEARREPORT', 'LENGTHREPORT', 'VALUE']].head(3)
                    print(f"    Sample data:")
                    for _, row in sample.iterrows():
                        print(f"      {row['YEARREPORT']}-Q{row['LENGTHREPORT']}: {row['VALUE']:,.0f}")
    
    # 5. Data completeness by year
    print(f"\n5. DATA COMPLETENESS BY YEAR (for IS.9 and IS.31):")
    print("-" * 50)
    
    years = sorted(combined_data['YEARREPORT'].dropna().unique())[-10:]  # Last 10 years
    
    for year in years:
        year_data = combined_data[combined_data['YEARREPORT'] == year]
        is9_brokers = year_data[year_data['KEYCODE'] == 'IS.9']['TICKER'].nunique()
        is31_brokers = year_data[year_data['KEYCODE'] == 'IS.31']['TICKER'].nunique()
        total_brokers_year = len([t for t in year_data['TICKER'].unique() if t != 'Sector'])
        
        print(f"{year}: IS.9 in {is9_brokers:2d}/{total_brokers_year:2d} brokers | IS.31 in {is31_brokers:2d}/{total_brokers_year:2d} brokers")
    
    return {
        'missing_is9': missing_is9,
        'missing_is31': missing_is31,
        'metric_availability': metric_availability
    }

if __name__ == "__main__":
    try:
        result = quick_analysis()
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()