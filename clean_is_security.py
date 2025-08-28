import csv
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
        if quarterly_records:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(quarterly_records)
    
    return quarterly_records

# Execute conversion
print("Converting IS_security from cumulative to quarterly format...")
quarterly_data = convert_cumulative_to_quarterly()

# Print sample data for verification
print(f"\nConverted {len(quarterly_data)} quarterly records")
print("\nSample data verification (OCSC 2017):")
ocsc_2017 = [row for row in quarterly_data if row['TICKER'] == 'OCSC' and row['YEARREPORT'] == '2017']
ocsc_2017.sort(key=lambda x: int(x['LENGTHREPORT']))

for row in ocsc_2017:
    quarter = f"Q{row['LENGTHREPORT']}"
    isa1_val = parse_float(row['ISA1'])
    isa1_str = f"{isa1_val:,.0f}" if isa1_val is not None else "N/A"
    print(f"{quarter} {row['YEARREPORT']}: {row['STARTDATE']} to {row['ENDDATE']} - ISA1: {isa1_str}")

print(f"\nData saved to sql/IS_security_quarterly.csv")