import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.brokerage_data import load_brokerage_metrics, get_available_tickers, get_available_quarters
from utils.investment_book import get_investment_data, format_investment_book, get_category_total


# Page config
st.set_page_config(page_title="Historical Financial Statements", layout="wide")

@st.cache_data(ttl=3600)
def load_broker_financial_data(ticker: str, start_year: int, end_year: int, include_annual: bool = True):
    """Load financial data for specific broker and year range (optimized query)"""
    try:
        # Load ONLY the selected broker and year range
        df = load_brokerage_metrics(ticker=ticker, start_year=start_year, end_year=end_year, include_annual=include_annual)
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
        ('NET_BROKERAGE_INCOME', 'Net Brokerage Income'),
        ('NET_IB_INCOME', 'Net IB Income'),
        ('FEE_INCOME', 'Fee Income'),
        ('NET_TRADING_INCOME', 'Net Trading Income'),
        ('INTEREST_INCOME', 'Interest Income'),
        ('NET_INVESTMENT_INCOME', 'Net Investment Income'),
        ('MARGIN_LENDING_INCOME', 'Margin Lending Income'),
        ('CAPITAL_INCOME', 'Capital Income'),
        ('TOTAL_OPERATING_INCOME', 'Total Operating Income'),
        ('SGA', 'SG&A'),
        ('INTEREST_EXPENSE', 'Interest Expense'),
        ('PBT', 'Profit Before Tax'),
        ('NPAT', 'Net Profit After Tax'),
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
        ('MARGIN_BALANCE', 'Margin Balance'),
        ('BORROWING_BALANCE', 'Borrowing Balance'),
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
    """Display Investment Book (Notes 7.x and 8.x) with FVTPL, AFS, HTM categories"""
    st.subheader("📊 Investment Book")
    st.markdown("Investment holdings classified by accounting category (FVTPL, AFS, HTM)")

    # Only show for quarterly data (investment book not meaningful for annual aggregates)
    quarterly_periods = [p for p in periods if p['LENGTHREPORT'] != 5]

    if not quarterly_periods:
        st.info("Investment book requires quarterly data. Please select a year range with quarterly periods.")
        return

    # Select which quarter to display (default to latest)
    period_labels = [p['QUARTER_LABEL'] for p in quarterly_periods]

    selected_period_label = st.selectbox(
        "Select Quarter for Investment Book:",
        period_labels,
        index=0,  # Latest quarter
        help="Investment holdings breakdown for the selected quarter"
    )

    # Find the selected period
    selected_period = next((p for p in quarterly_periods if p['QUARTER_LABEL'] == selected_period_label), None)

    if not selected_period:
        st.warning(f"Could not find data for {selected_period_label}")
        return

    year = selected_period['YEARREPORT']
    quarter = selected_period['LENGTHREPORT']

    # Get investment data
    investment_data = get_investment_data(df, broker, year, quarter)

    # Check if there's any investment data
    has_data = False
    for category in ['FVTPL', 'AFS', 'HTM']:
        if investment_data.get(category):
            if investment_data[category].get('Cost') or investment_data[category].get('Market Value'):
                has_data = True
                break

    if not has_data:
        st.info(f"No investment holdings data available for {broker} in {selected_period_label}")
        return

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fvtpl_mv = get_category_total(investment_data, 'FVTPL', 'Market Value') / 1_000_000_000
        st.metric("FVTPL (Trading)", f"{fvtpl_mv:,.1f}B")

    with col2:
        afs_mv = get_category_total(investment_data, 'AFS', 'Market Value') / 1_000_000_000
        st.metric("AFS", f"{afs_mv:,.1f}B")

    with col3:
        htm_cost = get_category_total(investment_data, 'HTM', 'Cost') / 1_000_000_000
        st.metric("HTM", f"{htm_cost:,.1f}B")

    with col4:
        total = fvtpl_mv + afs_mv + htm_cost
        st.metric("Total Investments", f"{total:,.1f}B")

    st.markdown("---")

    # Format and display investment book
    investment_df = format_investment_book(investment_data)

    if not investment_df.empty:
        st.dataframe(investment_df, use_container_width=True, hide_index=True, height=500)

        # Add download button
        csv = investment_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Investment Book",
            data=csv,
            file_name=f"investment_book_{broker}_{selected_period_label}.csv",
            mime="text/csv",
            key=f"download_inv_{broker}_{selected_period_label}"
        )

    # Add explanatory note
    with st.expander("ℹ️ Understanding Investment Categories"):
        st.markdown("""
        **FVTPL** (Fair Value Through Profit or Loss): Trading securities held for short-term profit

        **AFS** (Available-for-Sale): Long-term financial assets measured at fair value through OCI

        **HTM** (Held-to-Maturity): Fixed maturity investments measured at amortized cost

        **Columns:**
        - **Cost**: Original purchase price / carrying amount
        - **Market Value**: Current fair market value
        - **Unrealized G/L**: Difference between market value and cost
        """)


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
    available_brokers = ['Sector'] + individual_brokers

    selected_broker = st.sidebar.selectbox(
        "Select Broker:",
        options=available_brokers,
        index=0,
        help="Choose which broker's financial statements to display (Sector = sum of all brokers)"
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