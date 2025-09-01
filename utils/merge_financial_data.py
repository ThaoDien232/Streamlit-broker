import pandas as pd
import numpy as np
from datetime import datetime

def load_combined_financial_data():
    """Load the Combined_Financial_Data.csv file"""
    try:
        print("Loading Combined_Financial_Data.csv...")
        df = pd.read_csv('sql/Combined_Financial_Data.csv', low_memory=False)
        print(f"Loaded {len(df):,} records from Combined_Financial_Data.csv")
        return df
    except Exception as e:
        print(f"Error loading Combined_Financial_Data.csv: {e}")
        return pd.DataFrame()

def load_is_security_data():
    """Load the IS_security_calculated_metrics.csv file"""
    try:
        print("Loading IS_security_calculated_metrics.csv...")
        df = pd.read_csv('sql/IS_security_calculated_metrics.csv', low_memory=False)
        print(f"Loaded {len(df):,} records from IS_security_calculated_metrics.csv")
        return df
    except Exception as e:
        print(f"Error loading IS_security_calculated_metrics.csv: {e}")
        return pd.DataFrame()

def pivot_is_security_data(is_df):
    """
    Convert IS_security_calculated_metrics.csv from wide format to long format 
    to match Combined_Financial_Data structure
    """
    print("Pivoting IS_security_calculated_metrics to match Combined_Financial_Data format...")
    
    # Filter out rows with missing TICKER (empty rows)
    is_df = is_df[is_df['TICKER'].notna()].copy()
    
    # Define metric mapping with human-friendly names
    metric_mapping = {
        'FX_gain_loss': ('FX_GAIN_LOSS', 'FX gain/loss'),
        'Affiliates_divestment': ('AFFILIATES_DIVESTMENT', 'Gain/loss from affiliates divestment'),
        'Associate_income': ('ASSOCIATES_INCOME', 'Income from associate companies'),
        'Deposit_income': ('DEPOSIT_INCOME', 'Deposit income'),
        'Interest_expense': ('INTEREST_EXPENSE', 'Interest expense'),
        'Net_brokerage_income': ('NET_BROKERAGE_INCOME', 'Net Brokerage Income'),
        'Net_trading_income': ('NET_TRADING_INCOME', 'Net trading income'),
        'Interest_income': ('INTEREST_INCOME', 'Interest income'),
        'Net_investment_income': ('NET_INVESTMENT_INCOME', 'Net Investment Income'),
        'Net_other_operating_income': ('NET_OTHER_OP_INCOME', 'Net other operating income'),
        'Net_IB_income': ('NET_IB_INCOME', 'Net IB Income'),
        'Margin_lending_income': ('MARGIN_LENDING_INCOME', 'Margin Lending Income'),
        'Net_other_income': ('NET_OTHER_INCOME', 'Net other income'),
        'Fee_income': ('FEE_INCOME', 'Fee Income'),
        'Capital_income': ('CAPITAL_INCOME', 'Capital Income'),
        'Total_operating_income': ('TOTAL_OPERATING_INCOME', 'Total Operating Income'),
        'SGA': ('SGA', 'SG&A'),
        'PBT': ('PBT', 'PBT'),
        'NPAT': ('NPAT', 'NPAT'),
        'Operating_sales': ('OPERATING_SALES', 'Operating Sales'),
        'Cost_of_sales': ('COST_OF_SALES', 'Cost of Sales')
    }
    
    # Create records in Combined_Financial_Data format
    pivoted_records = []
    
    for _, row in is_df.iterrows():
        ticker = row['TICKER']
        year = int(row['Year']) if pd.notna(row['Year']) else 0
        quarter = int(row['Quarter']) if pd.notna(row['Quarter']) else 0
        start_date = row['STARTDATE'] if pd.notna(row['STARTDATE']) else ''
        end_date = row['ENDDATE'] if pd.notna(row['ENDDATE']) else ''
        quarter_label = f'Q{quarter}' if quarter in [1,2,3,4] else 'Annual'
        
        # Process each metric
        for original_col, (metric_code, metric_name) in metric_mapping.items():
            if original_col in row and pd.notna(row[original_col]):
                value = float(row[original_col])
                if value != 0:  # Only include non-zero values
                    pivoted_records.append({
                        'TICKER': ticker,
                        'YEARREPORT': year,
                        'LENGTHREPORT': quarter,
                        'STARTDATE': start_date,
                        'ENDDATE': end_date,
                        'NOTE': 'Calculated from IS security analysis',
                        'STATEMENT_TYPE': 'CALC',
                        'METRIC_CODE': metric_code,
                        'VALUE': value,
                        'KEYCODE': metric_code,
                        'KEYCODE_NAME': metric_name,
                        'QUARTER_LABEL': quarter_label
                    })
    
    pivoted_df = pd.DataFrame(pivoted_records)
    print(f"Created {len(pivoted_df):,} pivoted records from IS security data")
    return pivoted_df

def merge_financial_datasets(combined_df, pivoted_is_df):
    """Merge the pivoted IS security data with Combined_Financial_Data"""
    print("Merging IS security data with Combined_Financial_Data...")
    
    # Remove any existing CALC records from Combined_Financial_Data that might duplicate
    # the IS security calculated metrics to avoid duplicates
    existing_calc_metrics = set(pivoted_is_df['METRIC_CODE'].unique())
    
    # Filter out existing calculated metrics that would be duplicates
    combined_filtered = combined_df[
        ~((combined_df['STATEMENT_TYPE'] == 'CALC') & 
          (combined_df['METRIC_CODE'].isin(existing_calc_metrics)))
    ].copy()
    
    print(f"Removed {len(combined_df) - len(combined_filtered):,} existing calculated records to avoid duplicates")
    
    # Combine the datasets
    merged_df = pd.concat([combined_filtered, pivoted_is_df], ignore_index=True)
    
    # Sort by ticker, year, quarter, statement type, and metric code
    merged_df = merged_df.sort_values([
        'TICKER', 'YEARREPORT', 'LENGTHREPORT', 
        'STATEMENT_TYPE', 'METRIC_CODE'
    ])
    
    print(f"Merged dataset contains {len(merged_df):,} total records")
    return merged_df

def save_merged_data(merged_df):
    """Save the merged dataset"""
    output_file = 'sql/Combined_Financial_Data.csv'
    
    # Create backup of original
    backup_file = 'sql/Combined_Financial_Data_backup.csv'
    try:
        original_df = pd.read_csv(output_file)
        original_df.to_csv(backup_file, index=False)
        print(f"Created backup: {backup_file}")
    except:
        pass
    
    # Save merged data
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")
    
    return output_file

def display_merge_summary(merged_df):
    """Display summary of the merged dataset"""
    print("\n" + "="*80)
    print("MERGED FINANCIAL DATA SUMMARY")
    print("="*80)
    
    print(f"Total records: {len(merged_df):,}")
    print(f"Total tickers: {merged_df['TICKER'].nunique()}")
    print(f"Year range: {merged_df['YEARREPORT'].min()} - {merged_df['YEARREPORT'].max()}")
    
    print("\nRecords by Statement Type:")
    statement_counts = merged_df['STATEMENT_TYPE'].value_counts()
    for stmt_type, count in statement_counts.items():
        print(f"  {stmt_type}: {count:,} records")
    
    # Show calculated metrics breakdown
    calc_metrics = merged_df[merged_df['STATEMENT_TYPE'] == 'CALC']
    if len(calc_metrics) > 0:
        print(f"\nTop 15 Calculated Metrics:")
        calc_counts = calc_metrics['KEYCODE_NAME'].value_counts().head(15)
        for metric, count in calc_counts.items():
            metric_name = metric[:55] + "..." if len(metric) > 55 else metric
            print(f"  {metric_name}: {count:,} records")

def display_sample_merged_data(merged_df, sample_ticker='AAS', sample_year=2020, sample_quarter=1):
    """Display sample data to verify the merge"""
    print("\n" + "="*100)
    print(f"SAMPLE MERGED DATA - {sample_ticker} {sample_year} Q{sample_quarter}")
    print("="*100)
    
    sample = merged_df[
        (merged_df['TICKER'] == sample_ticker) & 
        (merged_df['YEARREPORT'] == sample_year) & 
        (merged_df['LENGTHREPORT'] == sample_quarter) &
        (merged_df['STATEMENT_TYPE'] == 'CALC')
    ].sort_values('METRIC_CODE').head(10)
    
    if len(sample) == 0:
        print(f"No calculated metrics found for {sample_ticker} {sample_year} Q{sample_quarter}")
        return
    
    print(f"{'Metric Code':<25} {'Metric Name':<40} {'Value':<15}")
    print("-" * 80)
    
    for _, row in sample.iterrows():
        metric_code = row['METRIC_CODE']
        metric_name = str(row['KEYCODE_NAME'])[:39]
        value = row['VALUE']
        
        if abs(value) >= 1_000_000_000:
            value_str = f"{value/1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            value_str = f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            value_str = f"{value/1_000:.1f}K"
        else:
            value_str = f"{value:.0f}"
        
        print(f"{metric_code:<25} {metric_name:<40} {value_str:<15}")

def main():
    """Main execution function"""
    print("Starting merge of IS security calculated metrics with Combined Financial Data...")
    print("="*80)
    
    # Load datasets
    combined_df = load_combined_financial_data()
    is_df = load_is_security_data()
    
    if combined_df.empty:
        print("ERROR: Could not load Combined_Financial_Data.csv")
        return
    
    if is_df.empty:
        print("ERROR: Could not load IS_security_calculated_metrics.csv")
        return
    
    # Convert IS security data to Combined_Financial_Data format
    pivoted_is_df = pivot_is_security_data(is_df)
    
    if pivoted_is_df.empty:
        print("WARNING: No data was pivoted from IS security file")
        return
    
    # Merge the datasets
    merged_df = merge_financial_datasets(combined_df, pivoted_is_df)
    
    # Save merged data
    output_file = save_merged_data(merged_df)
    
    # Display summary and sample
    display_merge_summary(merged_df)
    display_sample_merged_data(merged_df)
    
    print(f"\n" + "="*80)
    print("SUCCESS! IS security calculated metrics merged with Combined Financial Data")
    print(f"Output saved to: {output_file}")
    print(f"Total records: {len(merged_df):,}")
    print("="*80)

if __name__ == "__main__":
    main()