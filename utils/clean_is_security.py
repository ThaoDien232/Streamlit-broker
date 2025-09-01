import csv
import pandas as pd
from datetime import datetime
from collections import defaultdict

def parse_float(value):
    """Convert string to float, return None if empty/invalid"""
    if not value or value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None

def get_metrics_display_name():
    """
    Load and return keycode mapping from IRIS_KEYCODE.csv file.
    Returns a dictionary mapping DWHCode, KeyCode to KeyCodeName.
    """
    try:
        # Load the IRIS_KEYCODE.csv file
        keycode_df = pd.read_csv('sql/IRIS_KEYCODE.csv')
        
        # Create mapping dictionaries
        dwh_to_name = {}
        keycode_to_name = {}
        
        for _, row in keycode_df.iterrows():
            # Map DWHCode to KeyCodeName (for BS and IS codes)
            if pd.notna(row['DWHCode']) and pd.notna(row['KeyCodeName']):
                dwh_to_name[row['DWHCode']] = row['KeyCodeName']
            
            # Map KeyCode to KeyCodeName (for other codes)
            if pd.notna(row['KeyCode']) and pd.notna(row['KeyCodeName']):
                keycode_to_name[row['KeyCode']] = row['KeyCodeName']
        
        return {
            'dwh_to_name': dwh_to_name,
            'keycode_to_name': keycode_to_name,
            'full_df': keycode_df
        }
    except Exception as e:
        print(f"Error loading IRIS_KEYCODE.csv: {e}")
        return {
            'dwh_to_name': {},
            'keycode_to_name': {},
            'full_df': pd.DataFrame()
        }

def convert_cumulative_to_quarterly():
    # Read the CSV data
    with open('sql/IS_security.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Get column headers and identify numeric columns
    headers = reader.fieldnames
    numeric_cols = [col for col in headers if col.startswith(('ISA', 'ISS'))]
    
    # Group data by ticker and year
    groups = defaultdict(list)
    for row in data:
        key = (row['TICKER'], int(row['YEARREPORT']))
        groups[key].append(row)
    
    quarterly_records = []
    
    for (ticker, year), group in groups.items():
        # Filter out rows with empty dates and sort by end date
        valid_rows = [row for row in group if row['ENDDATE'] and row['STARTDATE']]
        if not valid_rows:
            continue
            
        valid_rows.sort(key=lambda x: datetime.strptime(x['ENDDATE'], '%Y-%m-%d'))
        
        # Identify periods by end date - choose the record with highest ISA1 value for each period
        periods = {}
        for row in valid_rows:
            try:
                end_date = datetime.strptime(row['ENDDATE'], '%Y-%m-%d')
                end_month_day = end_date.strftime('%m-%d')
                current_isa1 = parse_float(row['ISA1']) or 0
                
                if end_month_day == '03-31':
                    if 'Q1' not in periods or current_isa1 > (parse_float(periods['Q1']['ISA1']) or 0):
                        periods['Q1'] = row
                elif end_month_day == '06-30':
                    if 'Q2' not in periods or current_isa1 > (parse_float(periods['Q2']['ISA1']) or 0):
                        periods['Q2'] = row
                elif end_month_day == '09-30':
                    if 'Q3' not in periods or current_isa1 > (parse_float(periods['Q3']['ISA1']) or 0):
                        periods['Q3'] = row
                elif end_month_day == '12-31':
                    if 'Q4' not in periods or current_isa1 > (parse_float(periods['Q4']['ISA1']) or 0):
                        periods['Q4'] = row
            except ValueError:
                continue
        
        # Convert cumulative to quarterly
        if 'Q1' in periods:
            # Q1 = 3M data as-is
            q1_row = periods['Q1'].copy()
            q1_row['LENGTHREPORT'] = '1'
            q1_row['STARTDATE'] = f'{year}-01-01'
            q1_row['ENDDATE'] = f'{year}-03-31'
            quarterly_records.append(q1_row)
        
        if 'Q2' in periods and 'Q1' in periods:
            # Q2 = 6M - 3M
            q2_row = periods['Q2'].copy()
            for col in numeric_cols:
                val_6m = parse_float(q2_row[col])
                val_3m = parse_float(periods['Q1'][col])
                if val_6m is not None and val_3m is not None:
                    q2_row[col] = str(val_6m - val_3m)
                elif val_6m is not None:
                    q2_row[col] = str(val_6m)
            q2_row['LENGTHREPORT'] = '2'
            q2_row['STARTDATE'] = f'{year}-04-01'
            q2_row['ENDDATE'] = f'{year}-06-30'
            quarterly_records.append(q2_row)
        
        if 'Q3' in periods and 'Q2' in periods:
            # Q3 = 9M - 6M
            q3_row = periods['Q3'].copy()
            for col in numeric_cols:
                val_9m = parse_float(q3_row[col])
                val_6m = parse_float(periods['Q2'][col])
                if val_9m is not None and val_6m is not None:
                    q3_row[col] = str(val_9m - val_6m)
                elif val_9m is not None:
                    q3_row[col] = str(val_9m)
            q3_row['LENGTHREPORT'] = '3'
            q3_row['STARTDATE'] = f'{year}-07-01'
            q3_row['ENDDATE'] = f'{year}-09-30'
            quarterly_records.append(q3_row)
        
        if 'Q4' in periods and 'Q3' in periods:
            # Q4 = 12M - 9M
            q4_row = periods['Q4'].copy()
            for col in numeric_cols:
                val_12m = parse_float(q4_row[col])
                val_9m = parse_float(periods['Q3'][col])
                if val_12m is not None and val_9m is not None:
                    q4_row[col] = str(val_12m - val_9m)
                elif val_12m is not None:
                    q4_row[col] = str(val_12m)
            q4_row['LENGTHREPORT'] = '4'
            q4_row['STARTDATE'] = f'{year}-10-01'
            q4_row['ENDDATE'] = f'{year}-12-31'
            quarterly_records.append(q4_row)
    
    # Write to new CSV
    with open('sql/IS_security_quarterly.csv', 'w', newline='', encoding='utf-8') as f:
        ssi_q2_2025 = [row for row in quarterly_records if row['TICKER'] == 'SSI' and row['YEARREPORT'] == '2025' and row['LENGTHREPORT'] == '2']
        if ssi_q2_2025:
            print("\nDEBUG: SSI Q2 2025 raw IS.10 and IS.33 values:")
            for row in ssi_q2_2025:
                is10 = row.get('ISA10', 'N/A')
                is33 = row.get('ISA33', 'N/A')
                print(f"IS.10: {is10}, IS.33: {is33}")
            print("\nDEBUG: All IS items for SSI Q2 2025:")
            for row in ssi_q2_2025:
                for key, value in row.items():
                    if key.startswith('ISA') or key.startswith('ISS'):
                        print(f"{key}: {value}")

    # Print all calculated IS items for SSI for all quarters in 2024 and 2025
    ssi_quarterlies = [row for row in quarterly_records if row['TICKER'] == 'SSI' and row['YEARREPORT'] in [2024, 2025]]
    if ssi_quarterlies:
        print("\nDEBUG: SSI 2024-2025 quarterlies, all IS items:")
        for row in ssi_quarterlies:
            print(f"Year: {row['YEARREPORT']}, Quarter: {row['LENGTHREPORT']}, Period: {row['STARTDATE']} to {row['ENDDATE']}")
            for key, value in row.items():
                if key.startswith('ISA') or key.startswith('ISS'):
                    print(f"  {key}: {value}")
            print("-")
        if quarterly_records:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(quarterly_records)
    
    return quarterly_records

def calculate_financial_metrics_quarterly(quarterly_data):
    """
    Calculate financial statement items using quarterly data and logic from utils/data.py.
    Converts quarterly records into calculated metrics by ticker/year/quarter.
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(quarterly_data)
    
    # Convert numeric columns
    numeric_cols = [col for col in df.columns if col.startswith(('ISA', 'ISS', 'BSA', 'BSS'))]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Year'] = df['YEARREPORT'].astype(int)
    df['Quarter'] = df['LENGTHREPORT'].astype(int)
    
    calculated_metrics = []
    
    # Group by ticker and year for calculations
    for (ticker, year), group in df.groupby(['TICKER', 'Year']):
        # Sort by quarter to ensure proper order
        group = group.sort_values('Quarter')
        
        for _, row in group.iterrows():
            metrics_row = {
                'TICKER': ticker,
                'Year': year,
                'Quarter': int(row['Quarter']),
                'STARTDATE': row['STARTDATE'],
                'ENDDATE': row['ENDDATE']
            }
            
            # FX gain/loss - IS.44 + IS.50
            fx_44 = row.get('ISS44', 0) if pd.notna(row.get('ISS44')) else 0
            fx_50 = row.get('ISS50', 0) if pd.notna(row.get('ISS50')) else 0
            metrics_row['FX_gain_loss'] = fx_44 + fx_50
            
            # Affiliates divestment - IS.46
            metrics_row['Affiliates_divestment'] = row.get('ISS46', 0) if pd.notna(row.get('ISS46')) else 0
            
            # Associates income - IS.47 + IS.55
            assoc_47 = row.get('ISS47', 0) if pd.notna(row.get('ISS47')) else 0
            assoc_55 = row.get('ISS155', 0) if pd.notna(row.get('ISS155')) else 0  # IS.55 maps to ISS155
            metrics_row['Associate_income'] = assoc_47 + assoc_55
            
            # Deposit income - IS.45
            metrics_row['Deposit_income'] = row.get('ISS45', 0) if pd.notna(row.get('ISS45')) else 0
            
            # Interest expense - IS.51
            metrics_row['Interest_expense'] = row.get('ISS151', 0) if pd.notna(row.get('ISS151')) else 0  # IS.51 maps to ISS151
            
            # Net Brokerage Income - use ISS118 (Fee Income from statement)
            metrics_row['Net_brokerage_income'] = row.get('ISS118', 0) if pd.notna(row.get('ISS118')) else 0
            
            # Trading Income - use ISS119 (Securities Trading Income)  
            trading_income = row.get('ISS119', 0) if pd.notna(row.get('ISS119')) else 0
            metrics_row['Net_trading_income'] = trading_income
            
            # Interest Income - use ISS116 (Interest Income)
            interest_income = row.get('ISS116', 0) if pd.notna(row.get('ISS116')) else 0
            metrics_row['Interest_income'] = interest_income
            
            # Net Investment Income = Trading + Interest
            metrics_row['Net_investment_income'] = trading_income + interest_income
            
            # Other Operating Income - use ISS120 (Other Operating Income)
            other_op_income = row.get('ISS120', 0) if pd.notna(row.get('ISS120')) else 0
            metrics_row['Net_other_operating_income'] = other_op_income
            
            # Investment Banking Income - estimated from other components
            metrics_row['Net_IB_income'] = 0  # May need mapping from detailed breakdown
            
            # Margin Lending Income - estimated
            metrics_row['Margin_lending_income'] = 0
            
            # Other Income - estimated
            metrics_row['Net_other_income'] = 0
            
            # Fee Income = Net Brokerage + Net IB + Other Operating
            metrics_row['Fee_income'] = metrics_row['Net_brokerage_income'] + metrics_row['Net_IB_income'] + metrics_row['Net_other_operating_income']
            
            # Capital Income = Trading + Interest + Margin Lending
            metrics_row['Capital_income'] = metrics_row['Net_trading_income'] + metrics_row['Interest_income'] + metrics_row['Margin_lending_income']
            
            # Total Operating Income = Capital + Fee (or use ISA1 directly)
            metrics_row['Total_operating_income'] = row.get('ISA1', 0) if pd.notna(row.get('ISA1')) else 0
            
            # SG&A - use operating expenses ISA4 as proxy
            sga_value = row.get('ISA4', 0) if pd.notna(row.get('ISA4')) else 0
            metrics_row['SGA'] = abs(sga_value) if sga_value < 0 else 0
            
            # PBT - use ISA10 (likely profit before tax)
            metrics_row['PBT'] = row.get('ISA10', 0) if pd.notna(row.get('ISA10')) else 0
            
            # NPAT - use ISA23 or ISA24 (likely net profit after tax)
            npat_23 = row.get('ISA23', 0) if pd.notna(row.get('ISA23')) else 0
            npat_24 = row.get('ISA24', 0) if pd.notna(row.get('ISA24')) else 0
            metrics_row['NPAT'] = npat_23 if npat_23 != 0 else npat_24
            
            # Operating Sales (Revenue) - ISA1
            metrics_row['Operating_sales'] = row.get('ISA1', 0) if pd.notna(row.get('ISA1')) else 0
            
            # Cost of Sales - ISA4
            metrics_row['Cost_of_sales'] = row.get('ISA4', 0) if pd.notna(row.get('ISA4')) else 0
            
            calculated_metrics.append(metrics_row)
    
    return calculated_metrics

def display_verification_table(quarterly_data, metrics_mapping, sample_tickers=['OCSC', 'VND', 'SSI'], sample_year='2017'):
    """
    Display quarterly data in a formatted table for verification.
    """
    print("\n" + "="*100)
    print("QUARTERLY DATA VERIFICATION TABLE")
    print("="*100)
    
    # Get sample data
    sample_data = [row for row in quarterly_data 
                   if row['TICKER'] in sample_tickers and row['YEARREPORT'] == sample_year]
    
    if not sample_data:
        print(f"No data found for tickers {sample_tickers} in year {sample_year}")
        return
    
    # Sort by ticker and quarter
    sample_data.sort(key=lambda x: (x['TICKER'], int(x['LENGTHREPORT'])))
    
    # Get some key metrics to display
    key_metrics = ['ISA1', 'ISA4', 'ISA65', 'ISA71']  # Revenue, EBITDA, PBT, Net Income
    
    # Create table header
    header = f"{'TICKER':<8} {'Quarter':<8} {'Period':<20}"
    for metric in key_metrics:
        metric_name = metrics_mapping['dwh_to_name'].get(metric, metric)
        # Truncate long names
        if len(metric_name) > 15:
            metric_name = metric_name[:12] + "..."
        header += f" {metric_name:>15}"
    
    print(header)
    print("-" * len(header))
    
    # Display data rows
    for row in sample_data:
        ticker = row['TICKER']
        quarter = f"Q{row['LENGTHREPORT']}"
        period = f"{row['STARTDATE']} to {row['ENDDATE']}"
        
        row_str = f"{ticker:<8} {quarter:<8} {period:<20}"
        
        for metric in key_metrics:
            val = parse_float(row.get(metric, ''))
            if val is not None:
                if abs(val) >= 1_000_000_000:  # Billion
                    val_str = f"{val/1_000_000_000:.1f}B"
                elif abs(val) >= 1_000_000:  # Million
                    val_str = f"{val/1_000_000:.1f}M"
                elif abs(val) >= 1_000:  # Thousand
                    val_str = f"{val/1_000:.1f}K"
                else:
                    val_str = f"{val:.1f}"
            else:
                val_str = "N/A"
            row_str += f" {val_str:>15}"
        
        print(row_str)
    
    print("\n" + "="*100)
    
    # Display keycode mapping sample
    print("KEYCODE MAPPING SAMPLE (first 10 entries):")
    print("="*100)
    print(f"{'DWHCode':<10} {'KeyCode':<25} {'KeyCodeName':<50}")
    print("-" * 85)
    
    mapping_df = metrics_mapping['full_df']
    sample_mappings = mapping_df[mapping_df['DWHCode'].notna()].head(10)
    
    for _, row in sample_mappings.iterrows():
        dwh_code = str(row['DWHCode']) if pd.notna(row['DWHCode']) else 'N/A'
        key_code = str(row['KeyCode']) if pd.notna(row['KeyCode']) else 'N/A'
        key_name = str(row['KeyCodeName']) if pd.notna(row['KeyCodeName']) else 'N/A'
        
        if len(key_name) > 45:
            key_name = key_name[:42] + "..."
        
        print(f"{dwh_code:<10} {key_code:<25} {key_name:<50}")

# Execute conversion
print("Converting IS_security from cumulative to quarterly format...")
quarterly_data = convert_cumulative_to_quarterly()

# Load keycode mapping
print("Loading keycode mapping...")
metrics_mapping = get_metrics_display_name()

# Calculate financial metrics using utils/data.py logic
print("Calculating financial metrics using utils/data.py logic...")
calculated_metrics = calculate_financial_metrics_quarterly(quarterly_data)

print(f"\nConverted {len(quarterly_data)} quarterly records")
print(f"Calculated {len(calculated_metrics)} metric records")
print(f"Loaded {len(metrics_mapping['dwh_to_name'])} DWHCode mappings")
print(f"Loaded {len(metrics_mapping['keycode_to_name'])} KeyCode mappings")

# Save calculated metrics for use in 2_Charts.py
calculated_df = pd.DataFrame(calculated_metrics)
calculated_df.to_csv('sql/IS_security_calculated_metrics.csv', index=False)

# Display financial metrics verification table
def display_financial_metrics_table(calculated_metrics, sample_tickers=['SSI', 'VND', 'OCSC'], sample_year=2017):
    """Display calculated financial metrics in a table for verification"""
    print("\n" + "="*120)
    print("CALCULATED FINANCIAL METRICS VERIFICATION TABLE")
    print("="*120)
    
    # Filter sample data
    sample_data = [row for row in calculated_metrics 
                   if row['TICKER'] in sample_tickers and row['Year'] == sample_year]
    
    if not sample_data:
        print(f"No calculated metrics found for {sample_tickers} in year {sample_year}")
        return
    
    # Sort by ticker and quarter
    sample_data.sort(key=lambda x: (x['TICKER'], x['Quarter']))
    
    # Key metrics to display
    key_metrics = ['Total_operating_income', 'Fee_income', 'Capital_income', 'PBT', 'NPAT']
    
    # Create table header
    header = f"{'TICKER':<8} {'Qtr':<4} {'Period':<20}"
    for metric in key_metrics:
        header += f" {metric.replace('_', ' ')[:12]:>15}"
    
    print(header)
    print("-" * len(header))
    
    # Display data rows
    for row in sample_data:
        ticker = row['TICKER']
        quarter = f"Q{row['Quarter']}"
        period = f"{row['STARTDATE'][:5]} to {row['ENDDATE'][-5:]}"
        
        row_str = f"{ticker:<8} {quarter:<4} {period:<20}"
        
        for metric in key_metrics:
            val = row.get(metric, 0)
            if abs(val) >= 1_000_000_000:  # Billion
                val_str = f"{val/1_000_000_000:.1f}B"
            elif abs(val) >= 1_000_000:  # Million
                val_str = f"{val/1_000_000:.1f}M"
            elif abs(val) >= 1_000:  # Thousand
                val_str = f"{val/1_000:.1f}K"
            else:
                val_str = f"{val:.0f}"
            row_str += f" {val_str:>15}"
        
        print(row_str)

display_financial_metrics_table(calculated_metrics)

# Display summary by year for key metrics
print("\n" + "="*100)
print("ANNUAL SUMMARY OF KEY METRICS (VND 2017-2019)")
print("="*100)

annual_summary = {}
for row in calculated_metrics:
    if row['TICKER'] == 'VND' and row['Year'] in [2017, 2018, 2019]:
        key = (row['Year'], row['Quarter'])
        annual_summary[key] = row

if annual_summary:
    years = sorted(set(key[0] for key in annual_summary.keys()))
    print(f"{'Year':<6} {'Q':<3} {'Tot Op Income':<15} {'Fee Income':<12} {'Cap Income':<12} {'PBT':<12} {'NPAT':<12}")
    print("-" * 75)
    
    for year in years:
        for quarter in [1, 2, 3, 4]:
            if (year, quarter) in annual_summary:
                row = annual_summary[(year, quarter)]
                tot_op = f"{row['Total_operating_income']/1_000_000:.0f}M"
                fee = f"{row['Fee_income']/1_000_000:.0f}M"
                cap = f"{row['Capital_income']/1_000_000:.0f}M"
                pbt = f"{row['PBT']/1_000_000:.0f}M" if row['PBT'] != 0 else "0"
                npat = f"{row['NPAT']/1_000_000:.0f}M" if row['NPAT'] != 0 else "0"
                print(f"{year:<6} {quarter:<3} {tot_op:<15} {fee:<12} {cap:<12} {pbt:<12} {npat:<12}")

print(f"\nData saved to:")
print(f"- sql/IS_security_quarterly.csv (raw quarterly data)")
print(f"- sql/IS_security_calculated_metrics.csv (calculated metrics for 2_Charts.py)")
print("\nThe calculated metrics file can now be used in 2_Charts.py for plotting financial metrics!")