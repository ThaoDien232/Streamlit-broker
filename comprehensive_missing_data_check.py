#!/usr/bin/env python3
"""
Comprehensive Missing Data Check
Analysis of ALL metrics in the combined data vs original files
"""

import pandas as pd
import numpy as np

def comprehensive_analysis():
    print("Loading Combined Financial Data...")
    
    combined_data = pd.read_csv('sql/Combined_Financial_Data.csv')
    print(f"Combined data: {len(combined_data)} records")
    
    # Get all unique tickers (excluding non-broker entries)
    all_tickers = sorted([t for t in combined_data['TICKER'].dropna().unique() if t not in ['Sector', 'VISecurities', 'VinaSecurities', 'VPBANKS']])
    print(f"\nTotal brokers in combined data: {len(all_tickers)}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MISSING DATA ANALYSIS - ALL METRICS")
    print("="*80)
    
    # 1. Get all unique keycodes and analyze their distribution
    all_keycodes = sorted(combined_data['KEYCODE'].dropna().unique())
    print(f"\nTotal unique keycodes in combined data: {len(all_keycodes)}")
    
    # Separate IS and BS metrics
    is_keycodes = sorted([k for k in all_keycodes if k.startswith('IS.')])
    bs_keycodes = sorted([k for k in all_keycodes if k.startswith('BS.')])
    other_keycodes = sorted([k for k in all_keycodes if not k.startswith(('IS.', 'BS.'))])
    
    print(f"IS keycodes: {len(is_keycodes)}")
    print(f"BS keycodes: {len(bs_keycodes)}")
    print(f"Other keycodes: {len(other_keycodes)}")
    
    # 2. Analyze IS metrics comprehensively
    print(f"\n" + "="*60)
    print("INCOME STATEMENT (IS) METRICS ANALYSIS")
    print("="*60)
    
    is_missing_data = []
    is_partial_data = []
    is_good_data = []
    
    for keycode in is_keycodes:
        brokers_with_metric = combined_data[combined_data['KEYCODE'] == keycode]['TICKER'].nunique()
        coverage_pct = (brokers_with_metric / len(all_tickers)) * 100
        
        total_records = len(combined_data[combined_data['KEYCODE'] == keycode])
        
        if brokers_with_metric == 0:
            is_missing_data.append((keycode, brokers_with_metric, total_records, coverage_pct))
        elif coverage_pct < 25:
            is_partial_data.append((keycode, brokers_with_metric, total_records, coverage_pct))
        else:
            is_good_data.append((keycode, brokers_with_metric, total_records, coverage_pct))
    
    # Sort by coverage percentage
    is_missing_data.sort(key=lambda x: x[1])
    is_partial_data.sort(key=lambda x: x[3])
    is_good_data.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nIS METRICS SEVERELY MISSING (0-5 brokers):")
    print("-" * 50)
    for keycode, brokers, records, pct in is_missing_data[:20]:  # Show top 20 most missing
        print(f"{keycode:8s}: {brokers:3d}/{len(all_tickers):3d} brokers ({pct:5.1f}%) - {records:5d} records")
    
    print(f"\nIS METRICS PARTIALLY AVAILABLE (6-25% coverage):")
    print("-" * 50)
    for keycode, brokers, records, pct in is_partial_data:
        print(f"{keycode:8s}: {brokers:3d}/{len(all_tickers):3d} brokers ({pct:5.1f}%) - {records:5d} records")
    
    print(f"\nIS METRICS WITH GOOD COVERAGE (>25% coverage):")
    print("-" * 50)
    for keycode, brokers, records, pct in is_good_data[:30]:  # Show top 30
        print(f"{keycode:8s}: {brokers:3d}/{len(all_tickers):3d} brokers ({pct:5.1f}%) - {records:5d} records")
    
    # 3. Analyze BS metrics comprehensively
    print(f"\n" + "="*60)
    print("BALANCE SHEET (BS) METRICS ANALYSIS")
    print("="*60)
    
    bs_missing_data = []
    bs_partial_data = []
    bs_good_data = []
    
    for keycode in bs_keycodes:
        brokers_with_metric = combined_data[combined_data['KEYCODE'] == keycode]['TICKER'].nunique()
        coverage_pct = (brokers_with_metric / len(all_tickers)) * 100
        total_records = len(combined_data[combined_data['KEYCODE'] == keycode])
        
        if brokers_with_metric == 0:
            bs_missing_data.append((keycode, brokers_with_metric, total_records, coverage_pct))
        elif coverage_pct < 25:
            bs_partial_data.append((keycode, brokers_with_metric, total_records, coverage_pct))
        else:
            bs_good_data.append((keycode, brokers_with_metric, total_records, coverage_pct))
    
    # Sort by coverage percentage
    bs_missing_data.sort(key=lambda x: x[1])
    bs_partial_data.sort(key=lambda x: x[3])
    bs_good_data.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nBS METRICS SEVERELY MISSING (0-5 brokers):")
    print("-" * 50)
    for keycode, brokers, records, pct in bs_missing_data[:20]:
        print(f"{keycode:8s}: {brokers:3d}/{len(all_tickers):3d} brokers ({pct:5.1f}%) - {records:5d} records")
    
    print(f"\nBS METRICS PARTIALLY AVAILABLE (6-25% coverage):")
    print("-" * 50)
    for keycode, brokers, records, pct in bs_partial_data:
        print(f"{keycode:8s}: {brokers:3d}/{len(all_tickers):3d} brokers ({pct:5.1f}%) - {records:5d} records")
    
    print(f"\nBS METRICS WITH GOOD COVERAGE (>25% coverage):")
    print("-" * 50)
    for keycode, brokers, records, pct in bs_good_data[:30]:
        print(f"{keycode:8s}: {brokers:3d}/{len(all_tickers):3d} brokers ({pct:5.1f}%) - {records:5d} records")
    
    # 4. Find completely missing sequential IS and BS keycodes
    print(f"\n" + "="*60)
    print("SEQUENTIAL KEYCODE GAP ANALYSIS")
    print("="*60)
    
    # Extract numeric parts of IS keycodes to find gaps
    is_numbers = []
    for keycode in is_keycodes:
        try:
            num = float(keycode.replace('IS.', ''))
            is_numbers.append(num)
        except:
            pass
    
    is_numbers.sort()
    
    print(f"\nIS KEYCODE GAPS (missing sequential numbers):")
    print("-" * 40)
    
    missing_is_gaps = []
    for i in range(int(min(is_numbers)), int(max(is_numbers)) + 1):
        if i not in is_numbers:
            potential_keycode = f"IS.{i}"
            missing_is_gaps.append(potential_keycode)
            print(f"Missing: {potential_keycode}")
    
    # Same for BS
    bs_numbers = []
    for keycode in bs_keycodes:
        try:
            num = float(keycode.replace('BS.', ''))
            bs_numbers.append(num)
        except:
            pass
    
    bs_numbers.sort()
    
    print(f"\nBS KEYCODE GAPS (missing sequential numbers):")
    print("-" * 40)
    
    missing_bs_gaps = []
    for i in range(int(min(bs_numbers)), int(max(bs_numbers)) + 1):
        if i not in bs_numbers:
            potential_keycode = f"BS.{i}"
            missing_bs_gaps.append(potential_keycode)
            print(f"Missing: {potential_keycode}")
    
    # 5. Critical missing data summary for major brokers
    print(f"\n" + "="*60)
    print("CRITICAL MISSING DATA FOR MAJOR BROKERS")
    print("="*60)
    
    major_brokers = ['SSI', 'VCI', 'VIX', 'HCM', 'TCBS', 'MBS', 'VDS', 'FPTS', 'AGR', 'CTS']
    critical_is_metrics = [k for k, b, r, p in is_missing_data + is_partial_data if p < 10]  # Less than 10% coverage
    
    for broker in major_brokers:
        if broker in all_tickers:
            print(f"\n{broker} - Missing Critical Metrics:")
            broker_data = combined_data[combined_data['TICKER'] == broker]
            broker_keycodes = set(broker_data['KEYCODE'].unique())
            
            missing_count = 0
            for keycode in critical_is_metrics[:20]:  # Check top 20 most critical
                if keycode not in broker_keycodes:
                    print(f"  Missing: {keycode}")
                    missing_count += 1
            
            if missing_count == 0:
                print("  No critical missing metrics found")
    
    # 6. Export comprehensive report
    print(f"\n" + "="*60)
    print("GENERATING DETAILED REPORT...")
    print("="*60)
    
    # Create comprehensive report DataFrame
    all_metrics_report = []
    
    for keycode in all_keycodes:
        if keycode.startswith(('IS.', 'BS.')):
            brokers_with_metric = combined_data[combined_data['KEYCODE'] == keycode]['TICKER'].nunique()
            total_records = len(combined_data[combined_data['KEYCODE'] == keycode])
            coverage_pct = (brokers_with_metric / len(all_tickers)) * 100
            
            # Get sample brokers with this metric
            sample_brokers = list(combined_data[combined_data['KEYCODE'] == keycode]['TICKER'].unique())[:5]
            
            all_metrics_report.append({
                'KEYCODE': keycode,
                'TYPE': 'IS' if keycode.startswith('IS.') else 'BS',
                'BROKERS_WITH_DATA': brokers_with_metric,
                'TOTAL_BROKERS': len(all_tickers),
                'COVERAGE_PCT': round(coverage_pct, 1),
                'TOTAL_RECORDS': total_records,
                'SAMPLE_BROKERS': ', '.join(sample_brokers),
                'STATUS': 'MISSING' if brokers_with_metric == 0 else 'PARTIAL' if coverage_pct < 25 else 'GOOD'
            })
    
    report_df = pd.DataFrame(all_metrics_report)
    report_df = report_df.sort_values(['TYPE', 'COVERAGE_PCT'])
    
    # Save comprehensive report
    report_df.to_csv('comprehensive_missing_data_report.csv', index=False)
    print(f"Detailed report saved to: comprehensive_missing_data_report.csv")
    
    print(f"\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total IS metrics analyzed: {len(is_keycodes)}")
    print(f"IS metrics with 0 brokers: {len(is_missing_data)}")
    print(f"IS metrics with <25% coverage: {len(is_partial_data)}")
    print(f"IS metrics with good coverage: {len(is_good_data)}")
    print(f"\nTotal BS metrics analyzed: {len(bs_keycodes)}")
    print(f"BS metrics with 0 brokers: {len(bs_missing_data)}")
    print(f"BS metrics with <25% coverage: {len(bs_partial_data)}")
    print(f"BS metrics with good coverage: {len(bs_good_data)}")
    
    return {
        'is_missing': is_missing_data,
        'is_partial': is_partial_data,
        'bs_missing': bs_missing_data,
        'bs_partial': bs_partial_data,
        'missing_gaps': missing_is_gaps + missing_bs_gaps,
        'report_df': report_df
    }

if __name__ == "__main__":
    try:
        result = comprehensive_analysis()
        print(f"\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()