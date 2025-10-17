import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.brokerage_data import load_brokerage_metrics, get_available_tickers, get_available_quarters
from utils.investment_book import get_investment_data, format_investment_book, get_category_total


# Page config
st.set_page_config(page_title="Historical Financial Statements", layout="wide")

def calculate_net_brokerage_fee(df):
    """Calculate Net Brokerage Fee (bps) = NET_BROKERAGE_INCOME / (NOS101 + NOS109) * 10000"""
    calculated_rows = []

    for (year, quarter) in df[['YEARREPORT', 'LENGTHREPORT']].drop_duplicates().values:
        period_data = df[(df['YEARREPORT'] == year) & (df['LENGTHREPORT'] == quarter)]

        # Try both METRIC_CODE and KEYCODE for calculated metrics
        net_brok = period_data[
            (period_data['METRIC_CODE'] == 'Net_Brokerage_Income') |
            (period_data['KEYCODE'] == 'Net_Brokerage_Income')
        ]
        nos101 = period_data[period_data['KEYCODE'] == 'NOS101']
        nos109 = period_data[period_data['KEYCODE'] == 'NOS109']

        if not net_brok.empty and not nos101.empty and not nos109.empty:
            total_trading = nos101['VALUE'].values[0] + nos109['VALUE'].values[0]
            if total_trading > 0:
                fee_bps = (net_brok['VALUE'].values[0] / total_trading) * 10000
                calculated_rows.append({
                    'TICKER': period_data.iloc[0]['TICKER'],
                    'YEARREPORT': year,
                    'LENGTHREPORT': quarter,
                    'KEYCODE': 'NET_BROKERAGE_FEE',
                    'METRIC_CODE': 'NET_BROKERAGE_FEE',
                    'QUARTER_LABEL': period_data.iloc[0].get('QUARTER_LABEL', f"{quarter}Q{year%100:02d}"),
                    'KEYCODE_NAME': 'Net Brokerage Fee (bps)',
                    'VALUE': fee_bps,
                    'STATEMENT_TYPE': 'CALC'
                })

    if calculated_rows:
        calc_df = pd.DataFrame(calculated_rows)
        df = pd.concat([df, calc_df], ignore_index=True)
    return df

def calculate_margin_lending_rate(df):
    """Calculate Margin Lending Rate (%) = MARGIN_LENDING_INCOME / Avg(MARGIN_BALANCE, 4Q) * 4 * 100"""
    calculated_rows = []
    periods = df[['YEARREPORT', 'LENGTHREPORT']].drop_duplicates().sort_values(['YEARREPORT', 'LENGTHREPORT'])

    for _, (year, quarter) in periods.iterrows():
        # Try both METRIC_CODE and KEYCODE for calculated metrics
        margin_income_row = df[
            (df['YEARREPORT'] == year) &
            (df['LENGTHREPORT'] == quarter) &
            ((df['METRIC_CODE'] == 'Net_Margin_lending_Income') |
             (df['KEYCODE'] == 'Net_Margin_lending_Income'))
        ]

        if margin_income_row.empty:
            continue

        margin_income = margin_income_row['VALUE'].values[0]

        # Get trailing 4 quarters of margin balance using METRIC_CODE
        margin_books = []
        for q_offset in range(4):
            q_num = quarter - q_offset
            y = year
            while q_num <= 0:
                q_num += 4
                y -= 1

            margin_book_row = df[
                (df['YEARREPORT'] == y) &
                (df['LENGTHREPORT'] == q_num) &
                ((df['METRIC_CODE'] == 'Margin_Lending_book') |
                 (df['KEYCODE'] == 'Margin_Lending_book'))
            ]

            if not margin_book_row.empty:
                book_value = margin_book_row['VALUE'].values[0]
                if book_value > 0:
                    margin_books.append(book_value)

        if len(margin_books) >= 2 and margin_income:
            avg_margin_book = sum(margin_books) / len(margin_books)
            margin_rate = (margin_income / avg_margin_book) * 4 * 100

            calculated_rows.append({
                'TICKER': margin_income_row.iloc[0]['TICKER'],
                'YEARREPORT': year,
                'LENGTHREPORT': quarter,
                'KEYCODE': 'MARGIN_LENDING_RATE',
                'METRIC_CODE': 'MARGIN_LENDING_RATE',
                'QUARTER_LABEL': margin_income_row.iloc[0].get('QUARTER_LABEL', f"{quarter}Q{year%100:02d}"),
                'KEYCODE_NAME': 'Margin Lending Rate (%)',
                'VALUE': margin_rate / 100,  # Store as decimal
                'STATEMENT_TYPE': 'CALC'
            })

    if calculated_rows:
        calc_df = pd.DataFrame(calculated_rows)
        df = pd.concat([df, calc_df], ignore_index=True)
    return df

@st.cache_data(ttl=3600)
def load_broker_financial_data(ticker: str, start_year: int, end_year: int, include_annual: bool = True):
    """Load financial data for specific broker and year range (optimized query)"""
    try:
        # Load ONLY the selected broker and year range
        df = load_brokerage_metrics(ticker=ticker, start_year=start_year, end_year=end_year, include_annual=include_annual)

        if df is not None and not df.empty:
            # Calculate derived metrics
            df = calculate_net_brokerage_fee(df)
            df = calculate_margin_lending_rate(df)

        return df
    except Exception as e:
        st.error(f"Failed to load financial data from database: {e}")
        return None

def reload_data():
    """Clear cache and reload data"""
    load_broker_financial_data.clear()
    get_available_tickers.clear()
    st.rerun()

def format_vnd_billions(value):
    """Format numbers in VND billions with thousand separators"""
    if pd.isna(value) or value == 0:
        return "-"
    
    # Convert to billions
    billions = value / 1_000_000_000
    
    # Format with thousand separators and 1 decimal place
    if abs(billions) >= 1000:
        return f"{billions:,.0f}"
    elif abs(billions) >= 1:
        return f"{billions:,.1f}"
    else:
        return f"{billions:.2f}"

def get_available_periods(df, ticker, report_type):
    """Get available reporting periods for a ticker"""
    if report_type == "Annual":
        length_filter = [5]
    else:  # Quarterly
        length_filter = [1, 2, 3, 4]
    
    broker_data = df[
        (df['TICKER'] == ticker) & 
        (df['LENGTHREPORT'].isin(length_filter))
    ]
    
    if len(broker_data) == 0:
        return []
    
    # Get unique year-quarter combinations
    periods = broker_data[['YEARREPORT', 'LENGTHREPORT', 'QUARTER_LABEL']].drop_duplicates()
    periods = periods.sort_values(['YEARREPORT', 'LENGTHREPORT'], ascending=[True, True])
    
    return periods.to_dict('records')

def get_metric_value(df, ticker, year, quarter, metric_code):
    """Get a specific metric value for a ticker/period"""
    result = df[
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter) &
        (df['METRIC_CODE'] == metric_code)
    ]
    
    if len(result) > 0:
        return result.iloc[0]['VALUE']
    return 0

def get_calc_metric_value(df, ticker, year, quarter, metric_code):
    """Get a specific calculated metric value from CALC statement type"""
    result = df[
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter) &
        (df['STATEMENT_TYPE'] == 'CALC') &
        (df['METRIC_CODE'] == metric_code)
    ]
    
    if len(result) > 0:
        return result.iloc[0]['VALUE']
    return 0

def display_income_statement(df, ticker, periods, display_mode):
    """Display Income Statement calculated metrics"""
    st.subheader("📊 Income Statement")
    
    # Define key calculated IS metrics to display
    is_calc_metrics = [
        ('Net_Brokerage_Income', 'Net Brokerage Income'),
        ('NET_BROKERAGE_FEE', 'Net Brokerage Fee (bps)'),
        ('Net_IB_Income', 'Net IB Income'),
        ('Net_Fee_Income', 'Fee Income'),
        ('Net_Trading_Income', 'Net Trading Income'),
        ('Net_Interest_Income', 'Interest Income'),
        ('Net_investment_income', 'Net Investment Income'),
        ('Net_Margin_lending_Income', 'Margin Lending Income'),
        ('MARGIN_LENDING_RATE', 'Margin Lending Rate (%)'),
        ('MARGIN_LENDING_SPREAD', 'Margin Lending Spread (%)'),
        ('Net_Capital_Income', 'Capital Income'),
        ('Total_Operating_Income', 'Total Operating Income'),
        ('SGA', 'SG&A'),
        ('INTEREST_EXPENSE', 'Interest Expense'),
        ('PBT', 'Profit Before Tax'),
        ('NPAT', 'Net Profit After Tax'),
        ('ROE', 'Return on Equity (%)'),
        ('ROA', 'Return on Assets (%)'),
        ('INTEREST_RATE', 'Interest Rate (%)'),
    ]
    
    if display_mode == "Absolute Values":
        # Create data structure for absolute values
        is_data = []
        
        for metric_code, metric_name in is_calc_metrics:
            row = {'Metric': metric_name}
            has_data = False

            for period in periods:
                year = period['YEARREPORT']
                quarter = period['LENGTHREPORT']
                label = f"{year} {period['QUARTER_LABEL']}"

                value = get_calc_metric_value(df, ticker, year, quarter, metric_code)

                # Format based on metric type
                if metric_code in ['ROE', 'ROA']:
                    # These are already percentages (e.g., 15.5 = 15.5%)
                    row[label] = f"{value:.2f}%" if value != 0 else "-"
                elif metric_code in ['INTEREST_RATE', 'MARGIN_LENDING_RATE', 'MARGIN_LENDING_SPREAD']:
                    # Rates are stored as decimal, convert to percentage
                    row[label] = f"{value * 100:.2f}%" if value != 0 else "-"
                elif metric_code == 'NET_BROKERAGE_FEE':
                    # Basis points
                    row[label] = f"{value:.2f} bps" if value != 0 else "-"
                else:
                    # Financial values in VND
                    row[label] = format_vnd_billions(value)

                if value != 0:
                    has_data = True
            
            if has_data:
                is_data.append(row)
        
        if is_data:
            is_df = pd.DataFrame(is_data)
            st.dataframe(is_df, use_container_width=True, hide_index=True)
        else:
            st.info("No calculated Income Statement metrics available for the selected periods.")
    
    else:  # Growth mode
        is_data = []
        
        for metric_code, metric_name in is_calc_metrics:
            # Get values for all periods
            period_values = {}
            for period in periods:
                year = period['YEARREPORT']
                quarter = period['LENGTHREPORT']
                period_key = (year, quarter)
                value = get_calc_metric_value(df, ticker, year, quarter, metric_code)
                period_values[period_key] = value
            
            # Calculate YoY and QoQ growth
            yoy_row = {'Metric': f"{metric_name} (YoY%)"}
            qoq_row = {'Metric': f"{metric_name} (QoQ%)"}
            has_growth_data = False
            
            for i, period in enumerate(periods):
                year = period['YEARREPORT']
                quarter = period['LENGTHREPORT']
                label = f"{year} {period['QUARTER_LABEL']}"
                current_value = period_values[(year, quarter)]
                
                # YoY Growth
                prev_year_value = period_values.get((year-1, quarter), 0)
                if prev_year_value != 0 and current_value != 0:
                    yoy_growth = ((current_value - prev_year_value) / abs(prev_year_value)) * 100
                    yoy_row[label] = f"{yoy_growth:+.1f}%"
                    has_growth_data = True
                else:
                    yoy_row[label] = "-"
                
                # QoQ Growth
                if i > 0:
                    prev_period = periods[i-1]
                    prev_value = period_values[(prev_period['YEARREPORT'], prev_period['LENGTHREPORT'])]
                    if prev_value != 0 and current_value != 0:
                        qoq_growth = ((current_value - prev_value) / abs(prev_value)) * 100
                        qoq_row[label] = f"{qoq_growth:+.1f}%"
                        has_growth_data = True
                    else:
                        qoq_row[label] = "-"
                else:
                    qoq_row[label] = "-"  # First period has no previous quarter
            
            if has_growth_data:
                is_data.append(yoy_row)
                is_data.append(qoq_row)
        
        if is_data:
            is_df = pd.DataFrame(is_data)
            st.dataframe(is_df, use_container_width=True, hide_index=True)
        else:
            st.info("No growth data available for Income Statement metrics.")

def display_balance_sheet(df, ticker, periods, display_mode):
    """Display Balance Sheet calculated metrics"""
    st.subheader("🏦 Balance Sheet")
    
    # Define key calculated BS metrics to display
    bs_calc_metrics = [
        ('Margin_Lending_book', 'Margin Balance'),
        ('Borrowing_Balance', 'Borrowing Balance'),
    ]
    
    if display_mode == "Absolute Values":
        bs_data = []
        
        for metric_code, metric_name in bs_calc_metrics:
            row = {'Metric': metric_name}
            has_data = False
            
            for period in periods:
                year = period['YEARREPORT']
                quarter = period['LENGTHREPORT']
                label = f"{year} {period['QUARTER_LABEL']}"
                
                value = get_calc_metric_value(df, ticker, year, quarter, metric_code)
                row[label] = format_vnd_billions(value)
                
                if value != 0:
                    has_data = True
            
            if has_data:
                bs_data.append(row)
        
        if bs_data:
            bs_df = pd.DataFrame(bs_data)
            st.dataframe(bs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No calculated Balance Sheet metrics available for the selected periods.")
    
    else:  # Growth mode
        bs_data = []
        
        for metric_code, metric_name in bs_calc_metrics:
            # Get values for all periods
            period_values = {}
            for period in periods:
                year = period['YEARREPORT']
                quarter = period['LENGTHREPORT']
                period_key = (year, quarter)
                value = get_calc_metric_value(df, ticker, year, quarter, metric_code)
                period_values[period_key] = value
            
            # Calculate YoY and QoQ growth
            yoy_row = {'Metric': f"{metric_name} (YoY%)"}
            qoq_row = {'Metric': f"{metric_name} (QoQ%)"}
            has_growth_data = False
            
            for i, period in enumerate(periods):
                year = period['YEARREPORT']
                quarter = period['LENGTHREPORT']
                label = f"{year} {period['QUARTER_LABEL']}"
                current_value = period_values[(year, quarter)]
                
                # YoY Growth
                prev_year_value = period_values.get((year-1, quarter), 0)
                if prev_year_value != 0 and current_value != 0:
                    yoy_growth = ((current_value - prev_year_value) / abs(prev_year_value)) * 100
                    yoy_row[label] = f"{yoy_growth:+.1f}%"
                    has_growth_data = True
                else:
                    yoy_row[label] = "-"
                
                # QoQ Growth
                if i > 0:
                    prev_period = periods[i-1]
                    prev_value = period_values[(prev_period['YEARREPORT'], prev_period['LENGTHREPORT'])]
                    if prev_value != 0 and current_value != 0:
                        qoq_growth = ((current_value - prev_value) / abs(prev_value)) * 100
                        qoq_row[label] = f"{qoq_growth:+.1f}%"
                        has_growth_data = True
                    else:
                        qoq_row[label] = "-"
                else:
                    qoq_row[label] = "-"
            
            if has_growth_data:
                bs_data.append(yoy_row)
                bs_data.append(qoq_row)
        
        if bs_data:
            bs_df = pd.DataFrame(bs_data)
            st.dataframe(bs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No growth data available for Balance Sheet metrics.")

def display_investment_book(df, broker, periods):
    """Display Simplified Investment Book showing 4 asset groups with market values across quarters"""
    st.subheader("Investment Book")

    # DEBUG: Always check 3Q25 data first
    st.write(f"🔍 **DEBUG: Checking investment data for {broker}**")
    
    # Check if 3Q25 exists in the periods
    q3_25_period = None
    for period in periods:
        if period.get('QUARTER_LABEL') == '3Q25':
            q3_25_period = period
            break
    
    if q3_25_period:
        year = q3_25_period['YEARREPORT']
        quarter = q3_25_period['LENGTHREPORT']
        
        st.write(f"📅 **Found 3Q25 period: Year {year}, Quarter {quarter}**")
        
        # Check what raw data exists for this period
        period_raw_data = df[
            (df['TICKER'] == broker) &
            (df['YEARREPORT'] == year) &
            (df['LENGTHREPORT'] == quarter)
        ].copy()
        
        st.write(f"📊 Total records for {broker} 3Q25: {len(period_raw_data)}")
        
        if len(period_raw_data) > 0:
            statement_types = period_raw_data['STATEMENT_TYPE'].value_counts()
            st.write(f"📋 Statement types: {statement_types.to_dict()}")
            
            # Check for investment-specific METRIC_CODEs
            investment_codes = [
                'mtm_equities_market_value',
                'not_mtm_equities_market_value', 
                'bonds_market_value',
                'cds_deposits_market_value'
            ]
            
            st.write("🔍 Investment METRIC_CODEs in 3Q25 data:")
            for code in investment_codes:
                matching_records = period_raw_data[period_raw_data['METRIC_CODE'] == code]
                if len(matching_records) > 0:
                    value = matching_records['VALUE'].iloc[0]
                    st.write(f"  ✅ {code}: {len(matching_records)} records, value: {value:,.0f}")
                else:
                    st.write(f"  ❌ {code}: 0 records")
            
            # Show sample METRIC_CODEs that do exist
            unique_codes = period_raw_data['METRIC_CODE'].unique()
            st.write(f"📝 Sample METRIC_CODEs available ({len(unique_codes)} total):")
            for code in sorted(unique_codes)[:15]:
                st.write(f"  - {code}")
            if len(unique_codes) > 15:
                st.write(f"  ... and {len(unique_codes) - 15} more")
        else:
            st.write("❌ No data found for 3Q25")
    else:
        st.write("❌ 3Q25 period not found in available periods")
        st.write(f"Available periods: {[p.get('QUARTER_LABEL', 'Unknown') for p in periods[:10]]}")

    # Only show for quarterly data (investment book not meaningful for annual aggregates)
    quarterly_periods = [p for p in periods if p['LENGTHREPORT'] != 5]

    if not quarterly_periods:
        st.info("Investment book requires quarterly data. Please select a year range with quarterly periods.")
        return

    # Get last 6 quarters (or all available if less than 6)
    display_periods = quarterly_periods[:6]

    # Build simplified investment book table with MV across quarters
    investment_rows = []

    # Check if we have any investment data
    has_investment_data = False
    for period in display_periods:
        year = period['YEARREPORT']
        quarter = period['LENGTHREPORT']
        quarter_label = period['QUARTER_LABEL']
        
        # DEBUG: Check specifically for 3Q25 data
        if quarter_label == '3Q25':
            st.write(f"🔍 **DEBUG: Checking 3Q25 investment data for {broker}**")
            
            # Check what raw data exists for this period
            period_raw_data = df[
                (df['TICKER'] == broker) &
                (df['YEARREPORT'] == year) &
                (df['LENGTHREPORT'] == quarter)
            ].copy()
            
            st.write(f"📊 Total records for {broker} 3Q25: {len(period_raw_data)}")
            
            if len(period_raw_data) > 0:
                statement_types = period_raw_data['STATEMENT_TYPE'].value_counts()
                st.write(f"📋 Statement types: {statement_types.to_dict()}")
                
                # Check for investment-specific METRIC_CODEs
                investment_codes = [
                    'mtm_equities_market_value',
                    'not_mtm_equities_market_value', 
                    'bonds_market_value',
                    'cds_deposits_market_value'
                ]
                
                st.write("🔍 Investment METRIC_CODEs in 3Q25 data:")
                for code in investment_codes:
                    count = len(period_raw_data[period_raw_data['METRIC_CODE'] == code])
                    if count > 0:
                        value = period_raw_data[period_raw_data['METRIC_CODE'] == code]['VALUE'].iloc[0]
                        st.write(f"  ✅ {code}: {count} records, value: {value:,.0f}")
                    else:
                        st.write(f"  ❌ {code}: 0 records")
                
                # Show sample METRIC_CODEs that do exist
                unique_codes = period_raw_data['METRIC_CODE'].unique()
                st.write(f"📝 Sample METRIC_CODEs available ({len(unique_codes)} total):")
                for code in sorted(unique_codes)[:10]:
                    st.write(f"  - {code}")
                if len(unique_codes) > 10:
                    st.write(f"  ... and {len(unique_codes) - 10} more")
            else:
                st.write("❌ No data found for this period")
        
        period_data = get_investment_data(df, broker, year, quarter)
        
        # DEBUG: Show what get_investment_data returns for 3Q25
        if quarter_label == '3Q25':
            st.write(f"🔍 **DEBUG: get_investment_data result for 3Q25:**")
            for category, value in period_data.items():
                st.write(f"  - {category}: {value:,.0f}")
        
        if any(value > 0 for value in period_data.values()):
            has_investment_data = True
            break

    if not has_investment_data:
        st.info(f"No investment holdings data available for {broker}")
        return

    # Create table with 4 simplified asset groups
    from utils.investment_book import SIMPLIFIED_CATEGORIES
    
    for category in SIMPLIFIED_CATEGORIES:
        row = {'Asset Group': category}
        has_data = False

        for period in display_periods:
            year = period['YEARREPORT']
            quarter = period['LENGTHREPORT']
            label = period['QUARTER_LABEL']

            period_data = get_investment_data(df, broker, year, quarter)
            value = period_data.get(category, 0)
            
            # DEBUG: Show investment values for 3Q25
            if label == '3Q25':
                st.write(f"🔍 **DEBUG: {category} for 3Q25**: {value:,.0f} VND")

            if value != 0:
                row[label] = format_vnd_billions(value)
                has_data = True
            else:
                row[label] = '-'

        if has_data:
            investment_rows.append(row)
            # DEBUG: Show what rows are being added
            if '3Q25' in row:
                st.write(f"🔍 **DEBUG: Adding row for {category}**: {row}")

    # DEBUG: Show final investment_rows
    st.write(f"🔍 **DEBUG: Total investment rows created: {len(investment_rows)}**")
    if investment_rows:
        st.write("📋 **DEBUG: Investment rows summary:**")
        for i, row in enumerate(investment_rows):
            st.write(f"  Row {i+1}: {row['Asset Group']} - has 3Q25: {'3Q25' in row}")

    # Add total row
    total_row = {'Asset Group': 'TOTAL INVESTMENTS'}
    for period in display_periods:
        year = period['YEARREPORT']
        quarter = period['LENGTHREPORT']
        label = period['QUARTER_LABEL']

        period_data = get_investment_data(df, broker, year, quarter)
        total_value = sum(period_data.values())

        if total_value != 0:
            total_row[label] = format_vnd_billions(total_value)
        else:
            total_row[label] = '-'

    investment_rows.append(total_row)

    if investment_rows:
        investment_df = pd.DataFrame(investment_rows)
        
        # Style the dataframe to highlight the total row
        def highlight_total(row):
            return ['font-weight: bold' if row['Asset Group'] == 'TOTAL INVESTMENTS' else '' for _ in row]
        
        st.dataframe(
            investment_df.style.apply(highlight_total, axis=1),
            use_container_width=True, 
            hide_index=True
        )

def main():
    # Header with reload button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(" Historical Financial Statements")
    with col2:
        if st.sidebar.button("Reload Data", help="Refresh data from Combined_Financial_Data.csv"):
            reload_data()
    
    # Sidebar controls
    st.sidebar.header(" Display Options")

    # Display mode toggle
    display_mode = st.sidebar.radio(
        "Display Mode:",
        options=["Absolute Values", "Growth Rates"],
        index=0,
        help="Choose between absolute values in VND billions or YoY/QoQ growth percentages"
    )

    # Get available brokers (lightweight query)
    all_tickers = get_available_tickers()
    individual_brokers = sorted([t for t in all_tickers if t != 'Sector'])

    # Broker groups for organized display
    broker_groups = {
        'Top Tier': ['SSI', 'VCI', 'VND', 'HCM', 'TCBS', 'VPBS', 'VPS'],
        'Mid Tier': ['MBS', 'VIX', 'SHS', 'BSI', 'FTS'],
        'Regional': ['DSE', 'VDS', 'LPBS', 'Kafi', 'ACBS', 'OCBS', 'HDBS'],
    }

    # Create grouped broker list maintaining order within groups
    grouped_brokers = []
    for group_name, group_brokers in broker_groups.items():
        # Add brokers from this group that exist in individual_brokers
        for broker in group_brokers:
            if broker in individual_brokers:
                grouped_brokers.append(broker)

    # Add remaining brokers not in any group (sorted)
    remaining_brokers = sorted([b for b in individual_brokers if b not in grouped_brokers])
    all_brokers_ordered = ['Sector'] + grouped_brokers + remaining_brokers

    selected_broker = st.sidebar.selectbox(
        "Select Broker:",
        options=all_brokers_ordered,
        index=0,
        help="Brokers organized by tier: Top (SSI, VCI, VND, HCM, TCBS, VPBS, VPS) | Mid (MBS, VIX, SHS, BSI, FTS) | Regional (DSE, VDS, LPBS, Kafi, ACBS, OCBS, HDBS)"
    )

    # Report type selection
    report_type = st.sidebar.radio(
        "Report Type:",
        options=["Quarterly", "Annual"],
        index=0,
        help="Choose between quarterly or annual financial statements"
    )

    # Time horizon
    current_year = datetime.now().year
    available_years = list(range(2017, current_year + 1))
    available_years.sort(reverse=True)
    
    # Set default start and end years to 2024-2025
    default_start_year = 2024
    default_end_year = 2025
    
    # Find indices for default values, fallback to available data if not found
    try:
        default_start_index = available_years.index(default_start_year)
    except ValueError:
        # If 2024 not available, use the closest year or last available year
        default_start_index = min(5, len(available_years)-1) if len(available_years) > 5 else len(available_years)-1
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_year = st.selectbox(
            "From Year:",
            options=available_years,
            index=default_start_index,
            help="Starting year for the analysis"
        )
    
    with col2:
        # Filter end year options based on selected start year
        end_year_options = [year for year in available_years if year >= start_year]
        try:
            default_end_index = end_year_options.index(default_end_year) if default_end_year in end_year_options else 0
        except ValueError:
            default_end_index = 0
            
        end_year = st.selectbox(
            "To Year:",
            options=end_year_options,
            index=default_end_index,
            help="Ending year for the analysis"
        )
    
    # NOW load data ONLY for selected broker and year range
    include_annual = (report_type == "Annual")
    df = load_broker_financial_data(selected_broker, start_year, end_year, include_annual=True)

    if df is None or df.empty:
        st.warning(f"No data available for {selected_broker} in the selected time range.")
        return

    # Get available periods for selected broker
    periods = get_available_periods(df, selected_broker, report_type)

    # Filter periods by year range
    periods = [p for p in periods if start_year <= p['YEARREPORT'] <= end_year]

    if not periods:
        st.warning(f"No {report_type.lower()} data available for {selected_broker} in the selected time range.")
        return

    # Display broker info
    st.markdown(f"##  **{selected_broker}** Financial Statements")
    st.markdown(f"**Currency:** VND Billions")

    # Display all financial statements on one page
    display_income_statement(df, selected_broker, periods, display_mode)

    st.markdown("---")  # Separator line

    display_balance_sheet(df, selected_broker, periods, display_mode)

    st.markdown("---")  # Separator line

    display_investment_book(df, selected_broker, periods)

if __name__ == "__main__":
    main()