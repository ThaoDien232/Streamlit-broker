import pandas as pd
import numpy as np
from datetime import datetime

def load_keycode_mapping():
    """
    Load and return keycode mapping from IRIS_KEYCODE.csv file.
    Returns a dictionary mapping DWHCode to KeyCode and KeyCodeName.
    """
    try:
        keycode_df = pd.read_csv('sql/IRIS_KEYCODE.csv')
        
        # Create mapping dictionary: DWHCode -> KeyCode, KeyCodeName
        dwh_mapping = {}
        for _, row in keycode_df.iterrows():
            if pd.notna(row['DWHCode']) and pd.notna(row['KeyCode']):
                dwh_mapping[row['DWHCode']] = {
                    'KeyCode': row['KeyCode'],
                    'KeyCodeName': row['KeyCodeName'] if pd.notna(row['KeyCodeName']) else row['KeyCode']
                }
        
        return dwh_mapping
    except Exception as e:
        print(f"Error loading IRIS_KEYCODE.csv: {e}")
        return {}

def filter_and_combine_data():
    """
    Combine IS_security, BS_security, and Note_security files with specifications:
    1. Filter quarterly data (LENGTHREPORT = 1, 2, 3, 4, 5)
    2. Remove unnecessary columns
    3. Add statement type column
    4. Match DWHCode with KeyCode using IRIS_KEYCODE.csv
    """
    
    print("Loading keycode mapping...")
    keycode_mapping = load_keycode_mapping()
    print(f"Loaded {len(keycode_mapping)} DWHCode mappings")
    
    # Columns to remove (common across all files)
    columns_to_remove = [
        'INCOMESTATEMENTID', 'BALANCESHEETID', 'NOTESECURITYID',
        'ORGANCODE', 'COMTYPECODE', 'SOURCENAME', 'PUBLICDATE', 
        'ISAUDIT', 'STATUS', 'CREATEDATE', 'UPDATEDATE', 'LENGTHSERIES',
        'W_DATASOURCE_NUM_ID', 'W_INSERT_DT', 'W_INTEGRATION_ID', 'W_BATCH_ID'
    ]
    
    combined_data = []
    
    # Process Income Statement
    print("\nProcessing IS_security.csv...")
    try:
        is_df = pd.read_csv('sql/IS_security.csv')
        
        # Filter quarterly data (1=Q1, 2=Q2, 3=Q3, 4=Q4, 5=Annual)
        is_filtered = is_df[is_df['LENGTHREPORT'].isin([1, 2, 3, 4, 5])].copy()
        
        # Remove unnecessary columns
        is_filtered = is_filtered.drop(columns=[col for col in columns_to_remove if col in is_filtered.columns], errors='ignore')
        
        # Add statement type column
        is_filtered['STATEMENT_TYPE'] = 'IS'
        
        print(f"  Filtered {len(is_filtered)} IS records from {len(is_df)} total records")
        
        # Process each row and add keycode mapping
        for _, row in is_filtered.iterrows():
            # Get metric columns (ISA*, ISS*)
            metric_cols = [col for col in row.index if col.startswith(('ISA', 'ISS')) and pd.notna(row[col]) and row[col] != 0]
            
            for metric_col in metric_cols:
                # Create record for each metric
                record = {
                    'TICKER': row['TICKER'],
                    'YEARREPORT': row['YEARREPORT'],
                    'LENGTHREPORT': row['LENGTHREPORT'],
                    'STARTDATE': row['STARTDATE'],
                    'ENDDATE': row['ENDDATE'],
                    'NOTE': row.get('NOTE', ''),
                    'STATEMENT_TYPE': 'IS',
                    'METRIC_CODE': metric_col,
                    'VALUE': row[metric_col]
                }
                
                # Add KeyCode and KeyCodeName if mapping exists
                if metric_col in keycode_mapping:
                    record['KEYCODE'] = keycode_mapping[metric_col]['KeyCode']
                    record['KEYCODE_NAME'] = keycode_mapping[metric_col]['KeyCodeName']
                else:
                    record['KEYCODE'] = metric_col
                    record['KEYCODE_NAME'] = metric_col
                
                combined_data.append(record)
        
    except Exception as e:
        print(f"Error processing IS_security.csv: {e}")
    
    # Process Balance Sheet
    print("\nProcessing BS_security.csv...")
    try:
        bs_df = pd.read_csv('sql/BS_security.csv')
        
        # Filter quarterly data
        bs_filtered = bs_df[bs_df['LENGTHREPORT'].isin([1, 2, 3, 4, 5])].copy()
        
        # Remove unnecessary columns
        bs_filtered = bs_filtered.drop(columns=[col for col in columns_to_remove if col in bs_filtered.columns], errors='ignore')
        
        print(f"  Filtered {len(bs_filtered)} BS records from {len(bs_df)} total records")
        
        # Process each row and add keycode mapping
        for _, row in bs_filtered.iterrows():
            # Get metric columns (BSA*, BSB*, BSI*, BSS*)
            metric_cols = [col for col in row.index if col.startswith(('BSA', 'BSB', 'BSI', 'BSS')) and pd.notna(row[col]) and row[col] != 0]
            
            for metric_col in metric_cols:
                # Create record for each metric
                record = {
                    'TICKER': row['TICKER'],
                    'YEARREPORT': row['YEARREPORT'],
                    'LENGTHREPORT': row['LENGTHREPORT'],
                    'STARTDATE': row['STARTDATE'],
                    'ENDDATE': row['ENDDATE'],
                    'NOTE': row.get('NOTE', ''),
                    'STATEMENT_TYPE': 'BS',
                    'METRIC_CODE': metric_col,
                    'VALUE': row[metric_col]
                }
                
                # Add KeyCode and KeyCodeName if mapping exists
                if metric_col in keycode_mapping:
                    record['KEYCODE'] = keycode_mapping[metric_col]['KeyCode']
                    record['KEYCODE_NAME'] = keycode_mapping[metric_col]['KeyCodeName']
                else:
                    record['KEYCODE'] = metric_col
                    record['KEYCODE_NAME'] = metric_col
                
                combined_data.append(record)
        
    except Exception as e:
        print(f"Error processing BS_security.csv: {e}")
    
    # Process Notes
    print("\nProcessing Note_security.csv...")
    try:
        note_df = pd.read_csv('sql/Note_security.csv')
        
        # Filter quarterly data
        note_filtered = note_df[note_df['LENGTHREPORT'].isin([1, 2, 3, 4, 5])].copy()
        
        # Remove unnecessary columns
        note_filtered = note_filtered.drop(columns=[col for col in columns_to_remove if col in note_filtered.columns], errors='ignore')
        
        print(f"  Filtered {len(note_filtered)} Note records from {len(note_df)} total records")
        
        # Process each row and add keycode mapping
        for _, row in note_filtered.iterrows():
            # Get metric columns (NOS*)
            metric_cols = [col for col in row.index if col.startswith('NOS') and pd.notna(row[col]) and row[col] != 0]
            
            for metric_col in metric_cols:
                # Create record for each metric
                record = {
                    'TICKER': row['TICKER'],
                    'YEARREPORT': row['YEARREPORT'],
                    'LENGTHREPORT': row['LENGTHREPORT'],
                    'STARTDATE': row['STARTDATE'],
                    'ENDDATE': row['ENDDATE'],
                    'NOTE': row.get('NOTE', ''),
                    'STATEMENT_TYPE': 'Note',
                    'METRIC_CODE': metric_col,
                    'VALUE': row[metric_col]
                }
                
                # Add KeyCode and KeyCodeName if mapping exists
                if metric_col in keycode_mapping:
                    record['KEYCODE'] = keycode_mapping[metric_col]['KeyCode']
                    record['KEYCODE_NAME'] = keycode_mapping[metric_col]['KeyCodeName']
                else:
                    record['KEYCODE'] = metric_col
                    record['KEYCODE_NAME'] = metric_col
                
                combined_data.append(record)
        
    except Exception as e:
        print(f"Error processing Note_security.csv: {e}")
    
    return combined_data

def calculate_derived_metrics(combined_data):
    """
    Calculate derived financial metrics based on IS and BS data.
    This recreates the calculated metrics from utils/data.py
    """
    print("\nCalculating derived financial metrics...")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(combined_data)
    
    # Separate IS and BS data for calculations
    is_data = df[df['STATEMENT_TYPE'] == 'IS'].copy()
    bs_data = df[df['STATEMENT_TYPE'] == 'BS'].copy()
    
    derived_records = []
    
    # Get unique ticker-year-quarter combinations (limit to recent years for performance)
    periods = df[['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STARTDATE', 'ENDDATE']].drop_duplicates()
    periods = periods[periods['YEARREPORT'] >= 2020]  # Only calculate for 2020 onwards
    
    for _, period in periods.iterrows():
        ticker = period['TICKER']
        year = period['YEARREPORT']
        quarter = period['LENGTHREPORT']
        start_date = period['STARTDATE']
        end_date = period['ENDDATE']
        
        # Filter data for this specific period
        period_is = is_data[
            (is_data['TICKER'] == ticker) & 
            (is_data['YEARREPORT'] == year) & 
            (is_data['LENGTHREPORT'] == quarter)
        ]
        period_bs = bs_data[
            (bs_data['TICKER'] == ticker) & 
            (bs_data['YEARREPORT'] == year) & 
            (bs_data['LENGTHREPORT'] == quarter)
        ]
        
        if len(period_is) == 0 and len(period_bs) == 0:
            continue
        
        # Helper function to get metric value by KEYCODE (not METRIC_CODE)
        def get_metric_value_by_keycode(data, keycode):
            result = data[data['KEYCODE'] == keycode]
            return result['VALUE'].iloc[0] if len(result) > 0 else 0
        
        def get_metrics_sum_by_keycode(data, keycodes):
            return sum(get_metric_value_by_keycode(data, code) for code in keycodes)
        
        # Calculate derived metrics using correct IS.XX and BS.XX keycodes (matching calculate_new_metrics.py)
        
        # 1. FX gain/loss (IS.44 + IS.50)
        fx_gain_loss = get_metrics_sum_by_keycode(period_is, ['IS.44', 'IS.50'])
        
        # 2. Gain/loss from affiliates divestment (IS.46)
        affiliates = get_metric_value_by_keycode(period_is, 'IS.46')
        
        # 3. Income from associate companies (IS.47 + IS.55)
        associates = get_metrics_sum_by_keycode(period_is, ['IS.47', 'IS.55'])
        
        # 4. Deposit income (IS.45)
        deposit_inc = get_metric_value_by_keycode(period_is, 'IS.45')
        
        # 5. Interest expense (IS.51)
        interest_exp = get_metric_value_by_keycode(period_is, 'IS.51')
        
        # 6. Net IB Income
        ib_inc = get_metrics_sum_by_keycode(period_is, ['IS.12', 'IS.13', 'IS.15', 'IS.11', 'IS.16', 'IS.17', 'IS.18', 'IS.34', 'IS.35', 'IS.36', 'IS.38'])
        
        # 7. Net Brokerage Income (IS.10 + IS.33)
        net_brokerage = get_metrics_sum_by_keycode(period_is, ['IS.10', 'IS.33'])
        
        # 8. Net trading income
        trading_inc = get_metrics_sum_by_keycode(period_is, ['IS.3', 'IS.4', 'IS.5', 'IS.8', 'IS.9', 'IS.27', 'IS.24', 'IS.25', 'IS.26', 'IS.28', 'IS.29', 'IS.31', 'IS.32'])
        
        # 9. Interest income (IS.6)
        interest_inc = get_metric_value_by_keycode(period_is, 'IS.6')
        
        # 10. Net Investment Income
        net_investment_inc = trading_inc + interest_inc
        
        # 11. Margin Lending Income (IS.7 + IS.30)
        margin_inc = get_metrics_sum_by_keycode(period_is, ['IS.7', 'IS.30'])
        
        # 12. Net other income (IS.52 + IS.54 + IS.63)
        other_inc = get_metrics_sum_by_keycode(period_is, ['IS.52', 'IS.54', 'IS.63'])
        
        # 13. Net other operating income
        other_op_inc = get_metrics_sum_by_keycode(period_is, ['IS.14', 'IS.19', 'IS.20', 'IS.37', 'IS.39', 'IS.40'])
        
        # 14. Fee Income
        fee_income = net_brokerage + ib_inc + other_op_inc
        
        # 15. Capital Income
        capital_income = trading_inc + interest_inc + margin_inc
        
        # 16. Total Operating Income
        total_operating_income = capital_income + fee_income
        
        # 17. Borrowing Balance (BS.95 + BS.100 + BS.122 + BS.127)
        borrowing_balance = get_metrics_sum_by_keycode(period_bs, ['BS.95', 'BS.100', 'BS.122', 'BS.127'])
        
        # 18. SG&A (IS.57 + IS.58)
        sga = get_metrics_sum_by_keycode(period_is, ['IS.57', 'IS.58'])
        
        # 19. PBT (Profit Before Tax) - IS.65
        pbt = get_metric_value_by_keycode(period_is, 'IS.65')
        
        # 20. NPAT (Net Profit After Tax) - IS.71
        npat = get_metric_value_by_keycode(period_is, 'IS.71')
        
        # 21. Margin Balance (BS.8)
        margin_balance = get_metric_value_by_keycode(period_bs, 'BS.8')
        
        # Create derived metric records (matching original format)
        derived_metrics = [
            ('FX_GAIN_LOSS', 'FX Gain/Loss', fx_gain_loss),
            ('AFFILIATES', 'Gain/Loss from Affiliates Divestment', affiliates),
            ('ASSOCIATES', 'Income from Associate Companies', associates),
            ('DEPOSIT_INCOME', 'Deposit Income', deposit_inc),
            ('INTEREST_EXPENSE', 'Interest Expense', interest_exp),
            ('NET_IB_INCOME', 'Net IB Income', ib_inc),
            ('NET_BROKERAGE_INCOME', 'Net Brokerage Income', net_brokerage),
            ('NET_TRADING_INCOME', 'Net Trading Income', trading_inc),
            ('INTEREST_INCOME', 'Interest Income', interest_inc),
            ('NET_INVESTMENT_INCOME', 'Net Investment Income', net_investment_inc),
            ('MARGIN_LENDING_INCOME', 'Margin Lending Income', margin_inc),
            ('NET_OTHER_INCOME', 'Net Other Income', other_inc),
            ('NET_OTHER_OP_INCOME', 'Net Other Operating Income', other_op_inc),
            ('FEE_INCOME', 'Fee Income', fee_income),
            ('CAPITAL_INCOME', 'Capital Income', capital_income),
            ('TOTAL_OPERATING_INCOME', 'Total Operating Income', total_operating_income),
            ('BORROWING_BALANCE', 'Borrowing Balance', borrowing_balance),
            ('SGA', 'SG&A', sga),
            ('PBT', 'PBT', pbt),
            ('NPAT', 'NPAT', npat),
            ('MARGIN_BALANCE', 'Margin Balance', margin_balance),
        ]
        
        for metric_code, metric_name, value in derived_metrics:
            # Add all calculated metrics for every broker at every time period (including zeros)
            # This ensures complete data coverage as in the original implementation
            record = {
                'TICKER': ticker,
                'YEARREPORT': year,
                'LENGTHREPORT': quarter,
                'STARTDATE': start_date,
                'ENDDATE': end_date,
                'NOTE': 'Calculated from utils/data.py formulas',
                'STATEMENT_TYPE': 'CALC',
                'METRIC_CODE': metric_code,
                'VALUE': value,
                'KEYCODE': metric_code,
                'KEYCODE_NAME': metric_name
            }
            derived_records.append(record)
    
    print(f"  Generated {len(derived_records)} derived metric records")
    
    # Add derived metrics to combined data
    combined_data.extend(derived_records)
    
    return combined_data

def add_quarter_labels(df):
    """Add human-readable quarter labels"""
    def get_quarter_label(length_report):
        quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 5: 'Annual'}
        return quarter_map.get(length_report, f'Period_{length_report}')
    
    df['QUARTER_LABEL'] = df['LENGTHREPORT'].apply(get_quarter_label)
    return df

def display_summary_statistics(df):
    """Display summary statistics of the combined dataset"""
    print("\n" + "="*80)
    print("COMBINED FINANCIAL DATA SUMMARY")
    print("="*80)
    
    print(f"Total records: {len(df):,}")
    print(f"Total tickers: {df['TICKER'].nunique()}")
    print(f"Year range: {df['YEARREPORT'].min()} - {df['YEARREPORT'].max()}")
    
    print("\nRecords by Statement Type:")
    statement_counts = df['STATEMENT_TYPE'].value_counts()
    for stmt_type, count in statement_counts.items():
        print(f"  {stmt_type}: {count:,} records")
    
    print("\nRecords by Quarter:")
    quarter_counts = df['QUARTER_LABEL'].value_counts().sort_index()
    for quarter, count in quarter_counts.items():
        print(f"  {quarter}: {count:,} records")
    
    print("\nTop 10 Tickers by Record Count:")
    ticker_counts = df['TICKER'].value_counts().head(10)
    for ticker, count in ticker_counts.items():
        print(f"  {ticker}: {count:,} records")
    
    print("\nTop 10 Most Common Metrics:")
    metric_counts = df['KEYCODE_NAME'].value_counts().head(10)
    for metric, count in metric_counts.items():
        metric_name = metric[:50] + "..." if len(metric) > 50 else metric
        print(f"  {metric_name}: {count:,} records")

def calculate_sector_metrics(combined_data):
    """
    Calculate Sector metrics by summing all broker values for each metric in each period.
    Sector represents the aggregated value across all brokers for each metric.
    Uses YEARREPORT + LENGTHREPORT to identify the correct reporting period.
    """
    print("\nCalculating Sector metrics...")
    
    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame(combined_data)
    
    # Filter out records with NaN/empty tickers and exclude any existing 'Sector' records
    df_filtered = df[
        (df['TICKER'].notna()) & 
        (df['TICKER'] != '') & 
        (df['TICKER'] != 'Sector')
    ].copy()
    
    print(f"  Filtering from {len(df)} to {len(df_filtered)} records (excluding NaN/empty tickers)")
    
    # Group by the key reporting identifiers and aggregate
    sector_aggregates = df_filtered.groupby([
        'YEARREPORT', 'LENGTHREPORT', 
        'STATEMENT_TYPE', 'METRIC_CODE', 'KEYCODE', 'KEYCODE_NAME'
    ]).agg({
        'VALUE': 'sum',
        'STARTDATE': lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) > 0 else None,
        'ENDDATE': lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode()) > 0 else None
    }).reset_index()
    
    sector_data = []
    
    # Create sector records with proper time period classification
    for _, row in sector_aggregates.iterrows():
        year = row['YEARREPORT']
        length_report = row['LENGTHREPORT']
        
        # Generate proper STARTDATE and ENDDATE based on YEARREPORT and LENGTHREPORT
        if pd.isna(row['STARTDATE']) or pd.isna(row['ENDDATE']):
            if length_report == 1:  # Q1
                start_date = f"{year}-01-01"
                end_date = f"{year}-03-31"
            elif length_report == 2:  # Q2
                start_date = f"{year}-04-01"
                end_date = f"{year}-06-30"
            elif length_report == 3:  # Q3
                start_date = f"{year}-07-01"
                end_date = f"{year}-09-30"
            elif length_report == 4:  # Q4
                start_date = f"{year}-10-01"
                end_date = f"{year}-12-31"
            elif length_report == 5:  # Annual
                start_date = f"{year}-01-01"
                end_date = f"{year}-12-31"
            else:
                start_date = None
                end_date = None
        else:
            start_date = row['STARTDATE']
            end_date = row['ENDDATE']
        
        sector_record = {
            'TICKER': 'Sector',
            'YEARREPORT': year,
            'LENGTHREPORT': length_report,
            'STARTDATE': start_date,
            'ENDDATE': end_date,
            'NOTE': 'Sum of all brokers',
            'STATEMENT_TYPE': row['STATEMENT_TYPE'],
            'METRIC_CODE': row['METRIC_CODE'],
            'VALUE': row['VALUE'],
            'KEYCODE': row['KEYCODE'],
            'KEYCODE_NAME': row['KEYCODE_NAME']
        }
        sector_data.append(sector_record)
    
    print(f"  Generated {len(sector_data)} sector metric records")
    
    # Add sector data to the original combined data
    combined_data.extend(sector_data)
    
    return combined_data

def display_sample_data(df, sample_ticker='VND', sample_year=2019, sample_quarter=1):
    """Display sample data for verification"""
    print("\n" + "="*100)
    print(f"SAMPLE DATA - {sample_ticker} {sample_year} Q{sample_quarter}")
    print("="*100)
    
    sample = df[
        (df['TICKER'] == sample_ticker) & 
        (df['YEARREPORT'] == sample_year) & 
        (df['LENGTHREPORT'] == sample_quarter)
    ].head(20)
    
    if len(sample) == 0:
        print(f"No data found for {sample_ticker} {sample_year} Q{sample_quarter}")
        return
    
    print(f"{'Statement':<8} {'Metric Code':<12} {'KeyCode':<25} {'KeyCode Name':<40} {'Value':<15}")
    print("-" * 100)
    
    for _, row in sample.iterrows():
        stmt_type = row['STATEMENT_TYPE']
        metric_code = row['METRIC_CODE']
        keycode = row['KEYCODE'][:24] if len(str(row['KEYCODE'])) > 24 else str(row['KEYCODE'])
        keycode_name = str(row['KEYCODE_NAME'])[:39] if len(str(row['KEYCODE_NAME'])) > 39 else str(row['KEYCODE_NAME'])
        value = row['VALUE']
        
        if abs(value) >= 1_000_000_000:
            value_str = f"{value/1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            value_str = f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            value_str = f"{value/1_000:.1f}K"
        else:
            value_str = f"{value:.0f}"
        
        print(f"{stmt_type:<8} {metric_code:<12} {keycode:<25} {keycode_name:<40} {value_str:<15}")

# Execute the combination
if __name__ == "__main__":
    print("Starting financial data combination...")
    print("="*60)
    
    # Combine all data
    combined_data = filter_and_combine_data()
    
    if not combined_data:
        print("No data was processed. Check file paths and data availability.")
        exit(1)
    
    # Note: Derived metrics are calculated separately using utils/calculate_new_metrics.py
    
    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_data)
    
    # Add quarter labels
    combined_df = add_quarter_labels(combined_df)
    
    # Remove duplicates - keep the first occurrence of each unique combination
    print(f"\nRemoving duplicates from {len(combined_df):,} records...")
    
    # Define key columns that should be unique
    key_columns = ['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STATEMENT_TYPE', 'METRIC_CODE']
    
    # Check for duplicates before removal
    duplicates_before = combined_df.duplicated(subset=key_columns).sum()
    print(f"Found {duplicates_before:,} duplicate records")
    
    # Show sample of duplicated records for debugging
    if duplicates_before > 0:
        print("\nSample of duplicated records:")
        duplicate_mask = combined_df.duplicated(subset=key_columns, keep=False)
        sample_duplicates = combined_df[duplicate_mask].head(10)
        
        for _, row in sample_duplicates.iterrows():
            print(f"  {row['TICKER']} {row['YEARREPORT']} Q{row['LENGTHREPORT']} {row['STATEMENT_TYPE']} {row['METRIC_CODE']} = {row['VALUE']}")
    
    # Remove duplicates, keeping the first occurrence
    combined_df = combined_df.drop_duplicates(subset=key_columns, keep='first')
    
    # Report the result
    print(f"After deduplication: {len(combined_df):,} unique records remaining")
    
    # Calculate Sector metrics AFTER deduplication to ensure accuracy
    print("\nCalculating Sector metrics from deduplicated data...")
    combined_data = combined_df.to_dict('records')
    combined_data = calculate_sector_metrics(combined_data)
    combined_df = pd.DataFrame(combined_data)
    
    # Re-add quarter labels to ensure Sector data also has QUARTER_LABEL
    combined_df = add_quarter_labels(combined_df)
    
    # Sort by ticker, year, quarter, statement type
    combined_df = combined_df.sort_values(['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STATEMENT_TYPE', 'METRIC_CODE'])
    
    # Save to CSV
    output_file = 'sql/Combined_Financial_Data.csv'
    combined_df.to_csv(output_file, index=False)
    
    # Display statistics and sample data
    display_summary_statistics(combined_df)
    display_sample_data(combined_df)
    
    # Also display Sector sample data
    print("\n" + "="*100)
    print("SAMPLE SECTOR DATA - Aggregated values across all brokers")
    print("="*100)
    
    sector_sample = combined_df[combined_df['TICKER'] == 'Sector'].head(20)
    
    if len(sector_sample) > 0:
        print(f"{'Statement':<8} {'Metric Code':<12} {'KeyCode':<25} {'KeyCode Name':<40} {'Value':<15}")
        print("-" * 100)
        
        for _, row in sector_sample.iterrows():
            stmt_type = row['STATEMENT_TYPE']
            metric_code = row['METRIC_CODE']
            keycode = row['KEYCODE'][:24] if len(str(row['KEYCODE'])) > 24 else str(row['KEYCODE'])
            keycode_name = str(row['KEYCODE_NAME'])[:39] if len(str(row['KEYCODE_NAME'])) > 39 else str(row['KEYCODE_NAME'])
            value = row['VALUE']
            
            if abs(value) >= 1_000_000_000:
                value_str = f"{value/1_000_000_000:.1f}B"
            elif abs(value) >= 1_000_000:
                value_str = f"{value/1_000_000:.1f}M"
            elif abs(value) >= 1_000:
                value_str = f"{value/1_000:.1f}K"
            else:
                value_str = f"{value:.0f}"
            
            print(f"{stmt_type:<8} {metric_code:<12} {keycode:<25} {keycode_name:<40} {value_str:<15}")
    else:
        print("No Sector data found")
    
    print(f"\n" + "="*60)
    print(f"SUCCESS! Combined financial data saved to: {output_file}")
    print(f"Total records: {len(combined_df):,}")
    print("="*60)