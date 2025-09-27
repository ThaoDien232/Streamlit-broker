import pandas as pd
import numpy as np
import re

def parse_time_period(period_str):
    """
    Parse time period format like '5Q17', '1Q17' into YEARREPORT and LENGTHREPORT
    5Q17 means YEARREPORT 2017, LENGTHREPORT 5
    1Q17 means YEARREPORT 2017, LENGTHREPORT 1
    """
    if pd.isna(period_str) or period_str == '':
        return None, None
    
    # Extract quarter and year using regex
    match = re.match(r'(\d+)Q(\d+)', str(period_str))
    if match:
        quarter = int(match.group(1))
        year_short = int(match.group(2))
        
        # Convert 2-digit year to 4-digit year
        if year_short >= 0 and year_short <= 30:  # Assume 00-30 means 2000-2030
            year_full = 2000 + year_short
        else:  # Assume 31-99 means 1931-1999
            year_full = 1900 + year_short
        
        return year_full, quarter
    
    return None, None

def calculate_start_end_dates(year, length_report):
    """
    Calculate STARTDATE and ENDDATE based on quarter logic:
    1Q: YYYY-01-01 to YYYY-03-31
    2Q: YYYY-01-01 to YYYY-06-30
    3Q: YYYY-01-01 to YYYY-09-30
    4Q: YYYY-01-01 to YYYY-12-31
    5Q: YYYY-01-01 to YYYY-12-31 (annual)
    """
    start_date = f"{year}-01-01"
    
    if length_report == 1:
        end_date = f"{year}-03-31"
    elif length_report == 2:
        end_date = f"{year}-06-30"
    elif length_report == 3:
        end_date = f"{year}-09-30"
    elif length_report in [4, 5]:
        end_date = f"{year}-12-31"
    else:
        end_date = f"{year}-12-31"  # Default to annual
    
    return start_date, end_date

def transform_broker_data(excel_path):
    """
    Transform Broker sector.xlsx to match Combined_Financial_Data.csv format
    """
    print("Reading Broker sector.xlsx...")
    # Read the Excel file without headers
    df = pd.read_excel(excel_path, header=None)
    
    print(f"Original shape: {df.shape}")
    
    # Extract the key information
    # Row 0: KEYCODE_NAME (column headers)
    # Row 1: Time periods 
    # Column 0: Tickers (starting from row 2)
    
    keycode_names = df.iloc[0, 1:].tolist()  # Skip first column
    time_periods = df.iloc[1, 1:].tolist()   # Skip first column
    tickers = df.iloc[2:, 0].tolist()        # Skip first two rows
    
    # Get the data values (skip first column and first two rows)
    data_values = df.iloc[2:, 1:].values
    
    print(f"Found {len(tickers)} tickers and {len(keycode_names)} metrics")
    print(f"Sample tickers: {tickers[:5]}")
    print(f"Sample metrics: {set(keycode_names[:10])}")  # Show unique metrics
    print(f"Sample periods: {time_periods[:10]}")
    
    # Create the transformed data
    transformed_rows = []
    
    for ticker_idx, ticker in enumerate(tickers):
        if pd.isna(ticker) or ticker == '':
            continue
            
        for col_idx, (keycode_name, time_period) in enumerate(zip(keycode_names, time_periods)):
            if pd.isna(keycode_name) or keycode_name == '' or pd.isna(time_period):
                continue
                
            # Parse the time period
            year_report, length_report = parse_time_period(time_period)
            if year_report is None or length_report is None:
                continue
            
            # Get the value
            value = data_values[ticker_idx, col_idx]
            if pd.isna(value):
                continue
            
            # Calculate start and end dates
            start_date, end_date = calculate_start_end_dates(year_report, length_report)
            
            # Create quarter label
            if length_report == 5:
                quarter_label = 'Annual'
            else:
                quarter_label = f'{length_report}Q{str(year_report)[-2:]}'
            
            # Create the row in Combined_Financial_Data format
            row = {
                'TICKER': ticker,
                'YEARREPORT': year_report,
                'LENGTHREPORT': length_report,
                'STARTDATE': start_date,
                'ENDDATE': end_date,
                'NOTE': '',
                'STATEMENT_TYPE': 'BS',  # Assuming balance sheet data
                'METRIC_CODE': f'BROKER_{col_idx}',  # Generate unique metric code
                'VALUE': float(value),
                'KEYCODE': f'BROKER.{col_idx}',  # Generate unique keycode
                'KEYCODE_NAME': keycode_name,
                'QUARTER_LABEL': quarter_label
            }
            
            transformed_rows.append(row)
    
    print(f"Created {len(transformed_rows)} data rows")
    
    return pd.DataFrame(transformed_rows)

def filter_missing_data(new_df, existing_path):
    """
    Filter to only include data that doesn't already exist in Combined_Financial_Data.csv
    Only fill in missing tickers/periods, skip if data already exists for that combination
    """
    print("Loading existing Combined_Financial_Data.csv...")
    existing_df = pd.read_csv(existing_path)
    
    # Create a set of existing combinations (ticker, keycode_name, year, length)
    existing_combinations = set()
    for _, row in existing_df.iterrows():
        key = (row['TICKER'], row['KEYCODE_NAME'], row['YEARREPORT'], row['LENGTHREPORT'])
        existing_combinations.add(key)
    
    print(f"Existing combinations: {len(existing_combinations)}")
    print(f"New data combinations: {len(new_df)}")
    
    # Filter new_df to only include combinations that don't exist
    def is_missing_combination(row):
        key = (row['TICKER'], row['KEYCODE_NAME'], row['YEARREPORT'], row['LENGTHREPORT'])
        return key not in existing_combinations
    
    filtered_df = new_df[new_df.apply(is_missing_combination, axis=1)].copy()
    
    print(f"Missing combinations to add: {len(filtered_df)}")
    
    # Show some statistics
    if len(filtered_df) > 0:
        new_tickers = set(filtered_df['TICKER'].unique())
        existing_tickers = set(existing_df['TICKER'].unique())
        completely_new_tickers = new_tickers - existing_tickers
        
        print(f"Completely new tickers: {len(completely_new_tickers)} -> {sorted(list(completely_new_tickers))}")
        print(f"Existing tickers with new data: {len(new_tickers - completely_new_tickers)}")
    
    return filtered_df, existing_df

def merge_data(filtered_new_df, existing_df, output_path):
    """
    Merge the new data with existing Combined_Financial_Data.csv
    """
    print("Merging data...")
    
    # Combine the dataframes
    merged_df = pd.concat([existing_df, filtered_new_df], ignore_index=True)
    
    print(f"Final merged shape: {merged_df.shape}")
    
    # Sort by ticker, year, and length for better organization
    merged_df = merged_df.sort_values(['TICKER', 'YEARREPORT', 'LENGTHREPORT'])
    
    # Save the merged data
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")
    
    return merged_df

def main():
    """
    Main function to execute the merge process
    """
    try:
        # File paths
        broker_excel_path = 'Broker sector.xlsx'
        existing_csv_path = 'sql/Combined_Financial_Data.csv'
        output_csv_path = 'sql/Combined_Financial_Data.csv'  # Overwrite existing
        backup_path = 'sql/Combined_Financial_Data_backup.csv'
        
        # Create backup of original file
        print("Creating backup of original file...")
        existing_df = pd.read_csv(existing_csv_path)
        existing_df.to_csv(backup_path, index=False)
        print(f"Backup saved to {backup_path}")
        
        # Step 1: Transform broker data
        print("\n=== Step 1: Transform Broker Data ===")
        transformed_df = transform_broker_data(broker_excel_path)
        
        # Step 2: Filter missing data
        print("\n=== Step 2: Filter Missing Data ===")
        filtered_df, existing_df = filter_missing_data(transformed_df, existing_csv_path)
        
        if len(filtered_df) == 0:
            print("No new tickers to add. All tickers already exist in Combined_Financial_Data.csv")
            return
        
        # Step 3: Merge data
        print("\n=== Step 3: Merge Data ===")
        merged_df = merge_data(filtered_df, existing_df, output_csv_path)
        
        print("\n=== Summary ===")
        print(f"Original records: {len(existing_df)}")
        print(f"New records added: {len(filtered_df)}")
        print(f"Final records: {len(merged_df)}")
        print(f"New tickers added: {len(filtered_df['TICKER'].unique())}")
        
        # Show sample of new data
        print("\n=== Sample of New Data Added ===")
        sample_data = filtered_df.head(10)[['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'KEYCODE_NAME', 'VALUE']]
        print(sample_data.to_string(index=False))
        
    except Exception as e:
        print(f"Error during merge process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()