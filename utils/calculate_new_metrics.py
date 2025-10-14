import pandas as pd
import numpy as np
from datetime import datetime
from utils.keycode_matcher import load_keycode_map

def load_keycode_mapping():
    """Load the DWHCode to KeyCode mapping from IRIS_KEYCODE.csv"""
    try:
        keycode_map = load_keycode_map('sql/IRIS_KEYCODE.csv')
        print(f"Loaded {len(keycode_map)} keycode mappings")
        return keycode_map
    except Exception as e:
        print(f"Error loading keycode mapping: {e}")
        return {}

def load_combined_data():
    """Load the Combined_Financial_Data.csv file"""
    try:
        print("Loading Combined_Financial_Data.csv...")
        df = pd.read_csv('sql/Combined_Financial_Data.csv', low_memory=False)
        print(f"Loaded {len(df):,} records")
        return df
    except Exception as e:
        print(f"Error loading Combined_Financial_Data.csv: {e}")
        return pd.DataFrame()

def create_pivot_data(df):
    """Create pivot table for faster metric lookups"""
    print("Creating pivot table for faster processing...")
    
    # Filter only IS and BS data (exclude Note and CALC for now)
    source_data = df[df['STATEMENT_TYPE'].isin(['IS', 'BS'])].copy()
    
    # Create pivot table: index=(TICKER, YEARREPORT, LENGTHREPORT), columns=KEYCODE, values=VALUE
    # First standardize STARTDATE and ENDDATE to avoid pivot issues
    source_data['STARTDATE'] = source_data['STARTDATE'].fillna('')
    source_data['ENDDATE'] = source_data['ENDDATE'].fillna('')
    
    pivot_df = source_data.pivot_table(
        index=['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STARTDATE', 'ENDDATE'],
        columns='KEYCODE',
        values='VALUE',
        aggfunc='first',  # Take first value (duplicates should be removed in combine_financial_data.py)
        fill_value=0
    ).reset_index()
    
    print(f"Created pivot table with {len(pivot_df)} periods and {len(pivot_df.columns)-5} metrics")
    return pivot_df

def calculate_financial_metrics_vectorized(pivot_df, keycode_map):
    """
    Calculate all financial metrics using vectorized operations and correct keycode mapping.
    """
    print("Calculating financial metrics using vectorized operations...")
    
    # Load existing data to check what CALC records already exist
    existing_df = load_combined_data()
    existing_calc = existing_df[existing_df['STATEMENT_TYPE'] == 'CALC']
    existing_periods = set()
    if len(existing_calc) > 0:
        for _, row in existing_calc.iterrows():
            key = (row['TICKER'], row['YEARREPORT'], row['LENGTHREPORT'])
            existing_periods.add(key)
    
    print(f"Found {len(existing_periods)} periods with existing calculated metrics")
    
    calculated_records = []
    
    # Helper function to safely get column values using KeyCode
    def get_col(keycode):
        return pivot_df[keycode].fillna(0) if keycode in pivot_df.columns else pd.Series(0, index=pivot_df.index)
    
    # Helper function to get KeyCode from DWHCode (reverse mapping)
    reverse_keycode_map = {v: k for k, v in keycode_map.items()}
    def get_keycode_from_dwh(dwh_code):
        return reverse_keycode_map.get(dwh_code, dwh_code)
    
    # Process each row (ticker/year/quarter combination)
    for idx, row in pivot_df.iterrows():
        ticker = row['TICKER']
        year = row['YEARREPORT']
        quarter = row['LENGTHREPORT']
        start_date = row['STARTDATE'] if pd.notna(row['STARTDATE']) and row['STARTDATE'] != '' else ''
        end_date = row['ENDDATE'] if pd.notna(row['ENDDATE']) and row['ENDDATE'] != '' else ''
        # Format quarter label as "1Q25", "2Q25", etc.
        quarter_label = f'{quarter}Q{year % 100:02d}' if quarter in [1,2,3,4] else 'Annual'
        
        # Skip if this period already has calculated metrics
        period_key = (ticker, year, quarter)
        if period_key in existing_periods:
            continue
        
        # Skip Sector ticker - it will be calculated separately
        if ticker == 'Sector':
            continue
        
        # Calculate metrics using the formulas from utils/data.py
        metrics_to_calculate = []
        
        # Always calculate core metrics if we have the base IS/BS data for this period
        has_is_data = any(get_col(f'IS.{i}').iloc[idx] != 0 for i in range(1, 100) if f'IS.{i}' in pivot_df.columns)
        has_bs_data = any(get_col(f'BS.{i}').iloc[idx] != 0 for i in range(1, 200) if f'BS.{i}' in pivot_df.columns)
        
        # 1. FX gain/loss (IS.44 + IS.50) - Formula from IRIS_KEYCODE: IS.44+IS.50
        fx_gain_loss = get_col('IS.44').iloc[idx] + get_col('IS.50').iloc[idx]
        if fx_gain_loss != 0:
            metrics_to_calculate.append(('FX_GAIN_LOSS', 'FX gain/loss', fx_gain_loss))
        
        # 2. Affiliates divestment (IS.46) - Formula from IRIS_KEYCODE: IS.46
        affiliates = get_col('IS.46').iloc[idx]
        if affiliates != 0:
            metrics_to_calculate.append(('AFFILIATES_DIVESTMENT', 'Gain/loss from affiliates divestment', affiliates))
        
        # 3. Associates income (IS.47 + IS.55) - Formula from IRIS_KEYCODE: IS.55+IS.47
        associates = get_col('IS.47').iloc[idx] + get_col('IS.55').iloc[idx]
        if associates != 0:
            metrics_to_calculate.append(('ASSOCIATES_INCOME', 'Income from associate companies', associates))
        
        # 4. Deposit income (IS.45) - Formula from IRIS_KEYCODE: IS.45
        deposit_inc = get_col('IS.45').iloc[idx]
        if deposit_inc != 0:
            metrics_to_calculate.append(('DEPOSIT_INCOME', 'Deposit income', deposit_inc))
        
        # 5. Interest expense (IS.51) - Formula from IRIS_KEYCODE: IS.51
        interest_exp = get_col('IS.51').iloc[idx]
        if interest_exp != 0:
            metrics_to_calculate.append(('INTEREST_EXPENSE', 'Interest expense', interest_exp))
        
        # 6. Investment Banking Income (multiple IS codes)
        ib_codes = ['IS.12', 'IS.13', 'IS.15', 'IS.11', 'IS.16', 'IS.17', 'IS.18', 'IS.34', 'IS.35', 'IS.36', 'IS.38']
        ib_income = sum(get_col(code).iloc[idx] for code in ib_codes)
        if ib_income != 0:
            metrics_to_calculate.append(('NET_IB_INCOME', 'Net IB Income', ib_income))
        
        # 7. Net Brokerage Income (IS.10 + IS.33) - Revenue minus Expenses
        # IS.10 = Revenue in Brokerage services, IS.33 = Brokerage expenses (negative)
        is_10_revenue = get_col('IS.10').iloc[idx]  # Should be positive
        is_33_expenses = get_col('IS.33').iloc[idx]  # Should be negative
        net_brokerage = is_10_revenue + is_33_expenses  # This gives net (revenue - expenses)
        
        # Debug print for verification (especially SSI Q2 2025)
        if ticker == 'SSI' and year == 2025 and quarter == 2:
            print(f"DEBUG - {ticker} {year} Q{quarter}: IS.10={is_10_revenue/1e9:.3f}B + IS.33={is_33_expenses/1e9:.3f}B = {net_brokerage/1e9:.3f}B")
        
        if net_brokerage != 0 or (is_10_revenue != 0 and has_is_data):
            metrics_to_calculate.append(('NET_BROKERAGE_INCOME', 'Net Brokerage Income', net_brokerage))
        
        # 8. Net Trading Income - Formula from IRIS_KEYCODE: IS.3+IS.4+IS.5+IS.8+IS.9+IS.27+IS.24+IS.25+IS.26+IS.29+IS.28+IS.31+IS.32
        trading_codes = ['IS.3', 'IS.4', 'IS.5', 'IS.8', 'IS.9', 'IS.27', 'IS.24', 'IS.25', 'IS.26', 'IS.28', 'IS.29', 'IS.31', 'IS.32']
        trading_income = sum(get_col(code).iloc[idx] for code in trading_codes)
        if trading_income != 0:
            metrics_to_calculate.append(('NET_TRADING_INCOME', 'Net trading income', trading_income))
        
        # 9. Net Interest Income (IS.6) - Formula from IRIS_KEYCODE: IS.6
        interest_income = get_col('IS.6').iloc[idx]
        if interest_income != 0:
            metrics_to_calculate.append(('INTEREST_INCOME', 'Interest income', interest_income))
        
        # 10. Net Investment Income (Trading + Interest)
        net_investment_income = trading_income + interest_income
        if net_investment_income != 0:
            metrics_to_calculate.append(('NET_INVESTMENT_INCOME', 'Net Investment Income', net_investment_income))
        
        # 11. Net Margin Lending Income (IS.7 + IS.30) - Formula from IRIS_KEYCODE: IS.7+IS.30
        margin_lending = get_col('IS.7').iloc[idx] + get_col('IS.30').iloc[idx]
        if margin_lending != 0:
            metrics_to_calculate.append(('MARGIN_LENDING_INCOME', 'Margin Lending Income', margin_lending))
        
        # 12. Net Other Income - Formula from IRIS_KEYCODE: IS.54+IS.63+IS.52
        other_income = get_col('IS.52').iloc[idx] + get_col('IS.54').iloc[idx] + get_col('IS.63').iloc[idx]
        if other_income != 0:
            metrics_to_calculate.append(('NET_OTHER_INCOME', 'Net other income', other_income))
        
        # 13. Net Other Operating Income - Formula from IRIS_KEYCODE: IS.14+IS.19+IS.20+IS.37+IS.39+IS.40
        other_op_codes = ['IS.14', 'IS.19', 'IS.20', 'IS.37', 'IS.39', 'IS.40']
        other_op_income = sum(get_col(code).iloc[idx] for code in other_op_codes)
        if other_op_income != 0:
            metrics_to_calculate.append(('NET_OTHER_OP_INCOME', 'Net other operating income', other_op_income))
        
        # 14. Fee Income (Net Brokerage + Net IB + Other Operating)
        fee_income = net_brokerage + ib_income + other_op_income
        if fee_income != 0:
            metrics_to_calculate.append(('FEE_INCOME', 'Fee Income', fee_income))
        
        # 15. Capital Income (Trading + Interest + Margin Lending)
        capital_income = trading_income + interest_income + margin_lending
        if capital_income != 0:
            metrics_to_calculate.append(('CAPITAL_INCOME', 'Capital Income', capital_income))
        
        # 16. Total Operating Income (Capital + Fee)
        total_operating_income = capital_income + fee_income
        if total_operating_income != 0:
            metrics_to_calculate.append(('TOTAL_OPERATING_INCOME', 'Total Operating Income', total_operating_income))
        
        # 17. Borrowing Balance (BS.95 + BS.100 + BS.122 + BS.127)
        borrowing_balance = (get_col('BS.95').iloc[idx] + get_col('BS.100').iloc[idx] + 
                           get_col('BS.122').iloc[idx] + get_col('BS.127').iloc[idx])
        if borrowing_balance != 0:
            metrics_to_calculate.append(('BORROWING_BALANCE', 'Borrowing Balance', borrowing_balance))
        
        # 18. SG&A - Formula from IRIS_KEYCODE: IS.57+IS.58
        sga = get_col('IS.57').iloc[idx] + get_col('IS.58').iloc[idx]
        if sga != 0 or (get_col('IS.58').iloc[idx] != 0 and has_is_data):
            metrics_to_calculate.append(('SGA', 'SG&A', sga))
        
        # 19. PBT (IS.65) - Always calculate if we have IS.65
        pbt = get_col('IS.65').iloc[idx]
        if pbt != 0 or (get_col('IS.65').iloc[idx] != 0 or has_is_data):
            metrics_to_calculate.append(('PBT', 'PBT', pbt))
        
        # 20. NPAT (IS.71) - Always calculate if we have IS.71  
        npat = get_col('IS.71').iloc[idx]
        if npat != 0 or (get_col('IS.71').iloc[idx] != 0 or has_is_data):
            metrics_to_calculate.append(('NPAT', 'NPAT', npat))
        
        # 21. Margin Balance - Formula from IRIS_KEYCODE: BS.8 (Margin lending book)
        margin_balance = get_col('BS.8').iloc[idx]
        if margin_balance != 0:
            metrics_to_calculate.append(('MARGIN_BALANCE', 'Margin Balance', margin_balance))
        
        # 22. Total Assets - Formula from IRIS_KEYCODE: BS.92
        total_assets = get_col('BS.92').iloc[idx]
        if total_assets != 0:
            metrics_to_calculate.append(('TOTAL_ASSETS', 'Total Assets', total_assets))
        
        # 23. Total Equity - Formula from IRIS_KEYCODE: BS.142
        total_equity = get_col('BS.142').iloc[idx]
        if total_equity != 0:
            metrics_to_calculate.append(('TOTAL_EQUITY', 'Total Equity', total_equity))
        
        # 24. Net Margin Income - use the already calculated margin lending income (IS.7 + IS.30)
        # This is the same as the margin_lending calculation above, so we use that value
        net_margin_income = margin_lending  # margin_lending = IS.7 + IS.30 from line 164

        if net_margin_income != 0:
            metrics_to_calculate.append(('NET_MARGIN_INCOME', 'Net Margin Income', net_margin_income))

        # 25. Net Brokerage Fee (bps) - Net Brokerage Income / Total Trading Value * 10000
        # Formula: (NET_BROKERAGE_INCOME / (Institution_shares_trading_value + Investor_shares_trading_value)) * 10000
        institution_trading = get_col('Institution_shares_trading_value').iloc[idx]
        investor_trading = get_col('Investor_shares_trading_value').iloc[idx]
        total_trading_value = institution_trading + investor_trading

        if total_trading_value != 0 and net_brokerage != 0:
            # Calculate as basis points (bps)
            net_brokerage_fee_bps = (net_brokerage / total_trading_value) * 10000
            metrics_to_calculate.append(('NET_BROKERAGE_FEE_BPS', 'Net Brokerage Fee (bps)', net_brokerage_fee_bps))

        # Add all calculated metrics for this period
        for metric_code, metric_name, value in metrics_to_calculate:
            calculated_records.append({
                'TICKER': ticker,
                'YEARREPORT': year,
                'LENGTHREPORT': quarter,
                'STARTDATE': start_date,
                'ENDDATE': end_date,
                'NOTE': f'Calculated from utils/data.py formulas',
                'STATEMENT_TYPE': 'CALC',
                'METRIC_CODE': metric_code,
                'VALUE': float(value),
                'KEYCODE': metric_code,
                'KEYCODE_NAME': metric_name,
                'QUARTER_LABEL': quarter_label
            })
    
    print(f"Calculated {len(calculated_records):,} new metric records")
    return calculated_records

def calculate_sector_metrics(calculated_records):
    """
    Calculate Sector metrics by summing individual broker calculated metrics for each period.
    This ensures Sector calculated metrics follow the same logic as other Sector metrics.
    """
    print("Calculating Sector calculated metrics...")
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(calculated_records)
    
    if df.empty:
        return calculated_records
    
    # Group by period and metric, then sum across all brokers (excluding any existing Sector records)
    sector_aggregates = df[df['TICKER'] != 'Sector'].groupby([
        'YEARREPORT', 'LENGTHREPORT', 'METRIC_CODE', 'KEYCODE_NAME'
    ]).agg({
        'VALUE': 'sum',
        'STARTDATE': 'first',  # Use first occurrence
        'ENDDATE': 'first',    # Use first occurrence
        'NOTE': 'first',       # Use first occurrence
        'QUARTER_LABEL': 'first'  # Use first occurrence
    }).reset_index()
    
    sector_records = []
    
    for _, row in sector_aggregates.iterrows():
        sector_record = {
            'TICKER': 'Sector',
            'YEARREPORT': row['YEARREPORT'],
            'LENGTHREPORT': row['LENGTHREPORT'],
            'STARTDATE': row['STARTDATE'],
            'ENDDATE': row['ENDDATE'],
            'NOTE': 'Sum of all brokers - calculated metrics',
            'STATEMENT_TYPE': 'CALC',
            'METRIC_CODE': row['METRIC_CODE'],
            'VALUE': float(row['VALUE']),
            'KEYCODE': row['METRIC_CODE'],
            'KEYCODE_NAME': row['KEYCODE_NAME'],
            'QUARTER_LABEL': row['QUARTER_LABEL']
        }
        sector_records.append(sector_record)
    
    print(f"Calculated {len(sector_records):,} Sector calculated metric records")
    
    # Add sector records to the original list
    calculated_records.extend(sector_records)
    
    return calculated_records

def calculate_ratios(calculated_metrics):
    """
    Calculate Interest Rate and Margin Lending Rate ratios that require previous period data.
    """
    print("Calculating Interest Rate and Margin Lending Rate ratios...")

    # Load ALL calculated metrics from the database to have complete data for ratios
    combined_df = load_combined_data()
    calc_df = combined_df[combined_df['STATEMENT_TYPE'] == 'CALC'].copy()

    if calc_df.empty:
        print("No CALC metrics found in database")
        return calculated_metrics

    # Check which rates already exist
    existing_interest_rates = set()
    existing_margin_lending_rates = set()
    for _, row in calc_df.iterrows():
        key = (row['TICKER'], row['YEARREPORT'], row['LENGTHREPORT'])
        if row['METRIC_CODE'] == 'INTEREST_RATE':
            existing_interest_rates.add(key)
        elif row['METRIC_CODE'] == 'MARGIN_LENDING_RATE':
            existing_margin_lending_rates.add(key)

    print(f"Found {len(existing_interest_rates)} existing interest rate records")
    print(f"Found {len(existing_margin_lending_rates)} existing margin lending rate records")

    ratio_records = []

    # Group by ticker for ratio calculations
    for ticker in calc_df['TICKER'].unique():
        ticker_data = calc_df[calc_df['TICKER'] == ticker].sort_values(['YEARREPORT', 'LENGTHREPORT'])

        # Get NPAT, Total Equity, Total Assets, Interest Expense, Borrowing Balance, Margin Balance, and Net Margin Income by period
        npat_data = {}
        equity_data = {}
        assets_data = {}
        interest_expense_data = {}
        borrowing_balance_data = {}
        margin_balance_data = {}
        net_margin_income_data = {}

        for _, row in ticker_data.iterrows():
            key = (row['YEARREPORT'], row['LENGTHREPORT'])

            if row['METRIC_CODE'] == 'NPAT':
                npat_data[key] = row['VALUE']
            elif row['METRIC_CODE'] == 'TOTAL_EQUITY':
                equity_data[key] = row['VALUE']
            elif row['METRIC_CODE'] == 'TOTAL_ASSETS':
                assets_data[key] = row['VALUE']
            elif row['METRIC_CODE'] == 'INTEREST_EXPENSE':
                interest_expense_data[key] = row['VALUE']
            elif row['METRIC_CODE'] == 'BORROWING_BALANCE':
                borrowing_balance_data[key] = row['VALUE']
            elif row['METRIC_CODE'] == 'MARGIN_BALANCE':
                margin_balance_data[key] = row['VALUE']
            elif row['METRIC_CODE'] == 'NET_MARGIN_INCOME':
                net_margin_income_data[key] = row['VALUE']

        # Calculate ROE, ROA, Interest Rate, and Margin Lending Rate for each period
        periods = sorted(set(npat_data.keys()) | set(equity_data.keys()) | set(assets_data.keys()) |
                        set(interest_expense_data.keys()) | set(borrowing_balance_data.keys()) |
                        set(margin_balance_data.keys()) | set(net_margin_income_data.keys()))

        for i, period in enumerate(periods):
            year, quarter = period
            npat = npat_data.get(period, 0)
            current_equity = equity_data.get(period, 0)
            current_assets = assets_data.get(period, 0)
            interest_expense = interest_expense_data.get(period, 0)
            current_borrowing = borrowing_balance_data.get(period, 0)
            current_margin_balance = margin_balance_data.get(period, 0)
            net_margin_income = net_margin_income_data.get(period, 0)

            # Get previous period data
            if i > 0:
                prev_period = periods[i-1]
                prev_equity = equity_data.get(prev_period, current_equity)
                prev_assets = assets_data.get(prev_period, current_assets)
                prev_borrowing = borrowing_balance_data.get(prev_period, current_borrowing)
                prev_margin_balance = margin_balance_data.get(prev_period, current_margin_balance)
            else:
                prev_equity = current_equity
                prev_assets = current_assets
                prev_borrowing = current_borrowing
                prev_margin_balance = current_margin_balance

            # Skip ROE and ROA - they already exist
            # Only calculate new Interest Rate and Margin Lending Rate metrics

            period_key = (ticker, year, quarter)

            # Calculate Interest Rate (Interest Expense / Avg Borrowing Balance)
            # Only if it doesn't already exist
            if period_key not in existing_interest_rates:
                # Annualize for quarterly data: multiply by 4
                avg_borrowing = (current_borrowing + prev_borrowing) / 2 if prev_borrowing > 0 else current_borrowing
                if avg_borrowing > 0 and interest_expense != 0:
                    interest_rate = interest_expense / avg_borrowing
                    # Annualize quarterly data (multiply by 4), leave annual data as is
                    if quarter in [1, 2, 3, 4]:
                        interest_rate = interest_rate * 4

                    ratio_records.append({
                        'TICKER': ticker,
                        'YEARREPORT': year,
                        'LENGTHREPORT': quarter,
                        'STARTDATE': '',
                        'ENDDATE': '',
                        'NOTE': 'Calculated: Interest Expense / Avg Borrowing Balance (annualized for quarterly)',
                        'STATEMENT_TYPE': 'CALC',
                        'METRIC_CODE': 'INTEREST_RATE',
                        'VALUE': float(interest_rate),
                        'KEYCODE': 'INTEREST_RATE',
                        'KEYCODE_NAME': 'Interest Rate',
                        'QUARTER_LABEL': f'{quarter}Q{year % 100:02d}' if quarter in [1,2,3,4] else 'Annual'
                    })

            # Calculate Margin Lending Rate (Net Margin Income / Avg Margin Balance)
            # Only if it doesn't already exist
            if period_key not in existing_margin_lending_rates:
                # Annualize for quarterly data: multiply by 4
                avg_margin_balance = (current_margin_balance + prev_margin_balance) / 2 if prev_margin_balance > 0 else current_margin_balance
                if avg_margin_balance > 0 and net_margin_income != 0:
                    margin_lending_rate = net_margin_income / avg_margin_balance
                    # Annualize quarterly data (multiply by 4), leave annual data as is
                    if quarter in [1, 2, 3, 4]:
                        margin_lending_rate = margin_lending_rate * 4

                    ratio_records.append({
                        'TICKER': ticker,
                        'YEARREPORT': year,
                        'LENGTHREPORT': quarter,
                        'STARTDATE': '',
                        'ENDDATE': '',
                        'NOTE': 'Calculated: Net Margin Income / Avg Margin Balance (annualized for quarterly)',
                        'STATEMENT_TYPE': 'CALC',
                        'METRIC_CODE': 'MARGIN_LENDING_RATE',
                        'VALUE': float(margin_lending_rate),
                        'KEYCODE': 'MARGIN_LENDING_RATE',
                        'KEYCODE_NAME': 'Margin Lending Rate',
                        'QUARTER_LABEL': f'{quarter}Q{year % 100:02d}' if quarter in [1,2,3,4] else 'Annual'
                    })

    print(f"Calculated {len(ratio_records):,} rate records (Interest Rate + Margin Lending Rate)")

    # Add ratio records to the original list
    calculated_metrics.extend(ratio_records)

    return calculated_metrics

def calculate_net_brokerage_fee(calculated_metrics):
    """
    Calculate Net Brokerage Fee (bps) for all periods using trading value data from Note statements.
    Formula: (NET_BROKERAGE_INCOME / Total Trading Value) * 10000
    """
    print("Calculating Net Brokerage Fee (bps)...")

    # Load ALL data including Note data for trading values
    combined_df = load_combined_data()

    # Check which periods already have Net Brokerage Fee
    existing_fees = set()
    calc_df = combined_df[combined_df['STATEMENT_TYPE'] == 'CALC'].copy()
    for _, row in calc_df[calc_df['METRIC_CODE'] == 'NET_BROKERAGE_FEE_BPS'].iterrows():
        key = (row['TICKER'], row['YEARREPORT'], row['LENGTHREPORT'])
        existing_fees.add(key)

    print(f"Found {len(existing_fees)} existing net brokerage fee records")

    fee_records = []

    # Get unique tickers (excluding Sector for now)
    tickers = [t for t in calc_df['TICKER'].unique() if t != 'Sector']

    for ticker in tickers:
        # Get all periods for this ticker
        ticker_data = combined_df[combined_df['TICKER'] == ticker]

        # Get unique periods
        periods = ticker_data[['YEARREPORT', 'LENGTHREPORT']].drop_duplicates()

        for _, period_row in periods.iterrows():
            year = period_row['YEARREPORT']
            quarter = period_row['LENGTHREPORT']
            period_key = (ticker, year, quarter)

            # Skip if already calculated
            if period_key in existing_fees:
                continue

            # Get Net Brokerage Income from CALC
            net_brok_data = calc_df[
                (calc_df['TICKER'] == ticker) &
                (calc_df['YEARREPORT'] == year) &
                (calc_df['LENGTHREPORT'] == quarter) &
                (calc_df['METRIC_CODE'] == 'NET_BROKERAGE_INCOME')
            ]

            if net_brok_data.empty:
                continue

            net_brokerage_income = net_brok_data.iloc[0]['VALUE']

            # Get trading values from Note data
            institution_data = ticker_data[
                (ticker_data['YEARREPORT'] == year) &
                (ticker_data['LENGTHREPORT'] == quarter) &
                (ticker_data['KEYCODE'] == 'Institution_shares_trading_value')
            ]

            investor_data = ticker_data[
                (ticker_data['YEARREPORT'] == year) &
                (ticker_data['LENGTHREPORT'] == quarter) &
                (ticker_data['KEYCODE'] == 'Investor_shares_trading_value')
            ]

            if institution_data.empty or investor_data.empty:
                continue

            institution_trading = institution_data.iloc[0]['VALUE']
            investor_trading = investor_data.iloc[0]['VALUE']
            total_trading_value = institution_trading + investor_trading

            if total_trading_value > 0 and net_brokerage_income != 0:
                # Calculate as basis points (bps)
                net_brokerage_fee_bps = (net_brokerage_income / total_trading_value) * 10000

                fee_records.append({
                    'TICKER': ticker,
                    'YEARREPORT': year,
                    'LENGTHREPORT': quarter,
                    'STARTDATE': '',
                    'ENDDATE': '',
                    'NOTE': 'Calculated: Net Brokerage Income / Total Trading Value * 10000',
                    'STATEMENT_TYPE': 'CALC',
                    'METRIC_CODE': 'NET_BROKERAGE_FEE_BPS',
                    'VALUE': float(net_brokerage_fee_bps),
                    'KEYCODE': 'NET_BROKERAGE_FEE_BPS',
                    'KEYCODE_NAME': 'Net Brokerage Fee (bps)',
                    'QUARTER_LABEL': f'{quarter}Q{year % 100:02d}' if quarter in [1,2,3,4] else 'Annual'
                })

    print(f"Calculated {len(fee_records):,} Net Brokerage Fee records")

    # Add to calculated metrics
    calculated_metrics.extend(fee_records)

    return calculated_metrics

def append_to_combined_file(calculated_metrics):
    """Append calculated metrics to the Combined_Financial_Data.csv file."""
    if not calculated_metrics:
        print("No calculated metrics to append.")
        return
    
    print("Appending calculated metrics to Combined_Financial_Data.csv...")
    
    # Load existing data
    existing_df = load_combined_data()
    
    # Convert calculated metrics to DataFrame
    new_df = pd.DataFrame(calculated_metrics)
    
    # Combine with existing data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Sort by ticker, year, quarter, statement type
    combined_df = combined_df.sort_values(['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STATEMENT_TYPE', 'METRIC_CODE'])
    
    # Save back to file
    output_file = 'sql/Combined_Financial_Data.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"SUCCESS: Added {len(new_df):,} calculated metrics to {output_file}")
    print(f"Total records now: {len(combined_df):,}")
    
    return combined_df

def display_sample_data(df, sample_ticker='SSI', sample_year=2025, sample_quarter=2):
    """Display sample calculated metrics for verification"""
    print("\n" + "="*100)
    print(f"SAMPLE CALCULATED METRICS - {sample_ticker} {sample_year} Q{sample_quarter}")
    print("="*100)
    
    sample = df[
        (df['TICKER'] == sample_ticker) & 
        (df['YEARREPORT'] == sample_year) & 
        (df['LENGTHREPORT'] == sample_quarter) &
        (df['STATEMENT_TYPE'] == 'CALC')
    ].sort_values('METRIC_CODE')
    
    if len(sample) == 0:
        print(f"No calculated metrics found for {sample_ticker} {sample_year} Q{sample_quarter}")
        return
    
    print(f"{'Metric Code':<25} {'Metric Name':<35} {'Value':<15}")
    print("-" * 75)
    
    for _, row in sample.iterrows():
        metric_code = row['METRIC_CODE']
        metric_name = str(row['KEYCODE_NAME'])[:34]
        value = row['VALUE']
        
        if abs(value) >= 1_000_000_000:
            value_str = f"{value/1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            value_str = f"{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            value_str = f"{value/1_000:.1f}K"
        else:
            value_str = f"{value:.0f}"
        
        print(f"{metric_code:<25} {metric_name:<35} {value_str:<15}")

def display_summary_statistics(df):
    """Display summary statistics including calculated metrics"""
    print("\n" + "="*80)
    print("UPDATED COMBINED FINANCIAL DATA SUMMARY")
    print("="*80)
    
    print(f"Total records: {len(df):,}")
    print(f"Total tickers: {df['TICKER'].nunique()}")
    print(f"Year range: {df['YEARREPORT'].min()} - {df['YEARREPORT'].max()}")
    
    print("\nRecords by Statement Type:")
    statement_counts = df['STATEMENT_TYPE'].value_counts()
    for stmt_type, count in statement_counts.items():
        print(f"  {stmt_type}: {count:,} records")
    
    # Show calculated metrics breakdown
    calc_metrics = df[df['STATEMENT_TYPE'] == 'CALC']
    if len(calc_metrics) > 0:
        print(f"\nTop 10 Calculated Metrics:")
        calc_counts = calc_metrics['KEYCODE_NAME'].value_counts().head(10)
        for metric, count in calc_counts.items():
            metric_name = metric[:50] + "..." if len(metric) > 50 else metric
            print(f"  {metric_name}: {count:,} records")

# Execute the calculation
if __name__ == "__main__":
    print("Starting optimized calculation of new financial metrics...")
    print("="*70)
    
    # Load keycode mapping
    keycode_map = load_keycode_mapping()
    if not keycode_map:
        print("ERROR: Could not load keycode mapping. Exiting.")
        exit(1)
    
    # Load and process data
    df = load_combined_data()
    if df.empty:
        print("No data to process.")
        exit(1)
    
    # Create pivot table for faster processing
    pivot_df = create_pivot_data(df)
    
    # Calculate metrics using vectorized operations with keycode mapping
    calculated_metrics = calculate_financial_metrics_vectorized(pivot_df, keycode_map)
    
    # Calculate Sector metrics by summing individual broker metrics
    calculated_metrics = calculate_sector_metrics(calculated_metrics)
    
    # Calculate ROE and ROA ratios
    calculated_metrics = calculate_ratios(calculated_metrics)

    # Calculate Net Brokerage Fee (bps)
    calculated_metrics = calculate_net_brokerage_fee(calculated_metrics)

    if calculated_metrics:
        # Append to combined file
        updated_df = append_to_combined_file(calculated_metrics)
        
        # Display statistics and sample data
        display_summary_statistics(updated_df)
        display_sample_data(updated_df)
        
        print(f"\n" + "="*70)
        print("SUCCESS! New financial metrics calculated and added to Combined_Financial_Data.csv")
        print(f"Added {len(calculated_metrics):,} calculated metric records")
        print("="*70)
    else:
        print("No metrics were calculated. Check data availability and formulas.")