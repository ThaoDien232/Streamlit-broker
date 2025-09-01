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
    
    # Sort by ticker, year, quarter, statement type
    combined_df = combined_df.sort_values(['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STATEMENT_TYPE', 'METRIC_CODE'])
    
    # Save to CSV
    output_file = 'sql/Combined_Financial_Data.csv'
    combined_df.to_csv(output_file, index=False)
    
    # Display statistics and sample data
    display_summary_statistics(combined_df)
    display_sample_data(combined_df)
    
    print(f"\n" + "="*60)
    print(f"SUCCESS! Combined financial data saved to: {output_file}")
    print(f"Total records: {len(combined_df):,}")
    print("="*60)