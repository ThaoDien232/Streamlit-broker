import streamlit as st
import toml
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load theme from config.toml
theme_config = toml.load("utils/config.toml")
theme = theme_config["theme"]
primary_color = theme["primaryColor"]
background_color = theme["backgroundColor"]
secondary_background_color = theme["secondaryBackgroundColor"]
text_color = theme["textColor"]
font_family = theme["font"]

st.set_page_config(page_title="Financial Charts - CALC Metrics", layout="wide")

# OPTIMIZED: Load only selected data from database
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load_filtered_data(tickers, metrics, years, quarters):
    """Load only the selected tickers, metrics, years, and quarters"""
    from utils.brokerage_data import load_filtered_brokerage_data

    if not tickers or not metrics or not years or not quarters:
        return pd.DataFrame()

    df = load_filtered_brokerage_data(
        tickers=tickers,
        metrics=metrics,
        years=years,
        quarters=quarters
    )

    if df.empty:
        return pd.DataFrame()

    # Ensure correct data types
    df['YEARREPORT'] = df['YEARREPORT'].astype(int)
    df['LENGTHREPORT'] = df['LENGTHREPORT'].astype(int)
    return df

# Get metadata for filters (lightweight, cached 24 hours)
@st.cache_data(ttl=86400)
def get_available_options():
    """Get available tickers and years for filters - no heavy data loading"""
    from utils.brokerage_data import get_available_tickers_fast, get_available_years_fast

    tickers = get_available_tickers_fast()
    years = get_available_years_fast()

    return tickers, years

def reload_data():
    """Clear cache and reload data"""
    load_filtered_data.clear()
    get_available_options.clear()
    st.rerun()

@st.cache_data
def fetch_market_share_data(year, quarters):
    """Fetch brokerage market share data from HSX API"""
    all_data = []
    
    for quarter in quarters:
        try:
            url = f"https://api.hsx.vn/s/api/v1/1/brokeragemarketshare/top/ten"
            params = {
                'pageIndex': 1,
                'pageSize': 30,
                'year': year,
                'period': quarter,
                'dateType': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('success') and 'data' in data:
                brokerage_data = data['data'].get('brokerageStock', [])
                
                for item in brokerage_data:
                    all_data.append({
                        'Quarter': f"Q{quarter}",
                        'Year': year,
                        'Brokerage_Code': item.get('shortenName', ''),
                        'Brokerage_Name': item.get('name', ''),
                        'Market_Share_Percent': item.get('percentage', 0),
                        'Period_Label': item.get('period', f"Q{quarter}.{year}")
                    })
                    
        except Exception as e:
            st.warning(f"Failed to fetch data for Q{quarter} {year}: {e}")
            continue
    
    return pd.DataFrame(all_data)

def get_metric_display_name(metric_code):
    """Convert metric code to display name"""
    metric_names = {
        'NET_BROKERAGE_INCOME': 'Net Brokerage Income',
        'NET_TRADING_INCOME': 'Net Trading Income',
        'NET_INVESTMENT_INCOME': 'Net Investment Income',
        'FEE_INCOME': 'Fee Income',
        'CAPITAL_INCOME': 'Capital Income',
        'MARGIN_BALANCE': 'Margin Balance',
        'TOTAL_OPERATING_INCOME': 'Total Operating Income',
        'NET_IB_INCOME': 'Net IB Income',
        'NET_OTHER_OP_INCOME': 'Net Other Operating Income',
        'BORROWING_BALANCE': 'Borrowing Balance',
        'PBT': 'Profit Before Tax',
        'NPAT': 'Net Profit After Tax',
        'SGA': 'SG&A Expenses',
        'NET_MARGIN_INCOME': 'Net Margin Income',
        'MARGIN_LENDING_INCOME': 'Margin Lending Income',
        'ROE': 'Return on Equity (ROE)',
        'ROA': 'Return on Assets (ROA)'
    }
    return metric_names.get(metric_code, metric_code)

def format_value_billions(value):
    """Format value in billions with thousand separators"""
    if pd.isna(value) or value == 0:
        return "0.000"
    # Format with thousand separators
    return f"{value/1_000_000_000:,.0f}".replace(",", "{:,}").format(float(value/1_000_000_000))

def create_quarter_label(row):
    """Create quarter label like Q1 2017 or 2017 (Annual)"""
    if row['LENGTHREPORT'] == 5:
        return f"{int(row['YEARREPORT'])}"
    else:
        return f"Q{int(row['LENGTHREPORT'])} {int(row['YEARREPORT'])}"

# Header with reload button
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Financial Charts & Market Share")
with col2:
    if st.sidebar.button("Reload Data", help="Refresh data from Combined_Financial_Data.csv"):
        reload_data()

# Get available options (lightweight - no data loading)
available_brokers, available_years = get_available_options()

# Sidebar filters (set up BEFORE loading data)
st.sidebar.header("Chart Filters")

# Allowed metrics (all metrics available - no need to check database)
allowed_metrics = [
    'NET_BROKERAGE_INCOME',
    'NET_TRADING_INCOME',
    'NET_INVESTMENT_INCOME',
    'FEE_INCOME',
    'CAPITAL_INCOME',
    'MARGIN_BALANCE',
    'TOTAL_OPERATING_INCOME',
    'NET_IB_INCOME',
    'NET_OTHER_OP_INCOME',
    'BORROWING_BALANCE',
    'PBT',  # Changed from full name
    'NPAT',  # Changed from full name
    'SGA',
    'NET_MARGIN_INCOME',
    'MARGIN_LENDING_INCOME',
    'ROE',
    'ROA'
]

# Broker selection
selected_brokers = st.sidebar.multiselect(
    "Select Brokers:",
    options=available_brokers,
    default=available_brokers[:3] if len(available_brokers) >= 3 else available_brokers,
    key="chart_brokers"
)

# Metrics selection - NOW ALWAYS AVAILABLE
selected_metrics = st.sidebar.multiselect(
    "Select Metrics:",
    options=allowed_metrics,
    default=allowed_metrics[:3] if len(allowed_metrics) >= 3 else allowed_metrics,
    format_func=get_metric_display_name,
    key="chart_metrics"
)

# Year selection - from metadata
available_years_filtered = [year for year in available_years if 2021 <= year <= 2025]
selected_years = st.sidebar.multiselect(
    "Select Years:",
    options=available_years_filtered if available_years_filtered else available_years,
    default=available_years_filtered[-3:] if len(available_years_filtered) >= 3 else available_years_filtered,
    key="chart_years"
)

# Timeframe selection
timeframe_type = st.sidebar.radio(
    "Select Timeframe Type:",
    options=["Quarter", "Annual"],
    index=0,
    key="chart_timeframe"
)

# Quarter selection based on timeframe type
if timeframe_type == "Quarter":
    quarter_options = [1, 2, 3, 4]
    selected_quarters = st.sidebar.multiselect(
        "Select Quarters:",
        options=quarter_options,
        default=quarter_options,  # Default to all quarters
        format_func=lambda x: f"Q{x}",
        key="chart_quarters"
    )
else:  # Annual
    selected_quarters = [5]  # Annual data

# NOW load data based on selections (only what's needed)
df_calc = load_filtered_data(
    tickers=selected_brokers,
    metrics=selected_metrics,
    years=selected_years,
    quarters=selected_quarters
)

# Create quarter labels if data loaded
if not df_calc.empty:
    df_calc['Quarter_Label'] = df_calc.apply(create_quarter_label, axis=1)

# Create tabs for different sections
tab1, tab2 = st.tabs(["📊 Financial Charts", "📈 Market Share Data"])

with tab1:
    st.header("Financial Metrics Charts")
    
    if not df_calc.empty and selected_metrics and selected_brokers:
        # Data is already filtered by load_filtered_data, no need to filter again
        filtered_df = df_calc.copy()

        if not filtered_df.empty:
            
            # Display 2 charts per row using Streamlit columns
            for idx in range(0, len(selected_metrics), 2):
                # Create columns for current row
                if idx + 1 < len(selected_metrics):
                    col1, col2 = st.columns(2)
                    metrics_in_row = [selected_metrics[idx], selected_metrics[idx + 1]]
                    columns = [col1, col2]
                else:
                    col1, col2 = st.columns(2)
                    metrics_in_row = [selected_metrics[idx]]
                    columns = [col1]
                
                # Create charts for metrics in this row
                for i, metric in enumerate(metrics_in_row):
                    with columns[i]:
                        st.subheader(f"{get_metric_display_name(metric)}")

                        # DEBUG: Show what we're looking for
                        if metric in ['ROE', 'ROA']:
                            unique_metrics = filtered_df['METRIC_CODE'].unique()
                            st.info(f"DEBUG: Looking for '{metric}' in data. Available metrics: {unique_metrics[:10]}")
                            st.info(f"DEBUG: Total rows in filtered_df: {len(filtered_df)}")

                        # Filter data for current metric (METRIC_CODE is already the right format from database)
                        metric_data = filtered_df[filtered_df['METRIC_CODE'] == metric].copy()

                        # DEBUG: Show results
                        if metric in ['ROE', 'ROA']:
                            st.info(f"DEBUG: Found {len(metric_data)} rows for {metric}")
                            if not metric_data.empty:
                                st.write("Sample data:", metric_data.head())

                        if not metric_data.empty:
                            # Sort data chronologically
                            metric_data = metric_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])
                            
                            # Create bar chart
                            fig = go.Figure()
                            
                            # Color palette
                            colors = px.colors.qualitative.Set3
                            
                            # Add bar for each broker
                            for j, broker in enumerate(selected_brokers):
                                broker_data = metric_data[metric_data['TICKER'] == broker].copy()
                                
                                if not broker_data.empty:
                                    # Check if this is ROE or ROA (percentage metrics)
                                    if metric in ['ROE', 'ROA']:
                                        # For ROE/ROA, multiply by 100 to convert to percentage
                                        broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce') * 100
                                        
                                        # Annualize quarterly values by multiplying by 4
                                        # Only annualize if it's quarterly data (LENGTHREPORT != 5)
                                        quarterly_mask = broker_data['LENGTHREPORT'] != 5
                                        broker_data.loc[quarterly_mask, 'DISPLAY_VALUE'] *= 4
                                        
                                        y_values = broker_data['DISPLAY_VALUE']
                                        hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.2f}}%<br><extra></extra>"
                                    else:
                                        # Convert other values to billions for display
                                        broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce') / 1_000_000_000
                                        y_values = broker_data['DISPLAY_VALUE']
                                        hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.3f}}B VND<br><extra></extra>"
                                    
                                    # Sort by period to ensure consistent ordering
                                    broker_data = broker_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])
                                    
                                    fig.add_trace(
                                        go.Bar(
                                            x=broker_data['Quarter_Label'],
                                            y=y_values,
                                            name=broker,
                                            marker_color=colors[j % len(colors)],
                                            hovertemplate=hover_template
                                        )
                                    )
                            
                            # Set y-axis title and format based on metric type
                            if metric in ['ROE', 'ROA']:
                                yaxis_title = "Percentage (%)"
                                tick_format = ".2f"
                            else:
                                yaxis_title = "Value (Billions VND)"
                                tick_format = ".0f"
                            
                            # Update layout
                            fig.update_layout(
                                height=400,
                                title=f"{get_metric_display_name(metric)} - {timeframe_type} Comparison",
                                xaxis_title="Period",
                                yaxis_title=yaxis_title,
                                hovermode='x unified',
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                barmode='group'
                            )
                            fig.update_yaxes(tickformat=tick_format)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(f"No data available for {get_metric_display_name(metric)} with current filters.")
        else:
            st.warning("Please select brokers and metrics to display charts.")

with tab2:
    st.header("Brokerage Market Share Data")

    # Add dedicated filters for Market Share tab
    col_year, col_quarters = st.columns([1, 2])

    with col_year:
        # Year filter - available years from 2021 to 2025
        available_years = list(range(2021, 2026))
        ms_year = st.selectbox(
            "Select Year",
            options=available_years,
            index=len(available_years) - 1,  # Default to latest year (2025)
            key="market_share_year"
        )

    with col_quarters:
        # Quarter filter
        ms_quarters = st.multiselect(
            "Select Quarters",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}",
            key="market_share_quarters"
        )
    
    if ms_quarters:
        # Only fetch data if we have valid years (2021-2025 range)
        if 2021 <= ms_year <= 2025:
            with st.spinner("Fetching market share data..."):
                market_share_df = fetch_market_share_data(ms_year, ms_quarters)
        else:
            st.warning(f"No market share data available for year {ms_year}. Please select a year between 2021-2025.")
            market_share_df = pd.DataFrame()
        
        if not market_share_df.empty:
            st.subheader(f"Market Share Data for {ms_year}")
            
            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Brokers",
                    value=market_share_df['Brokerage_Code'].nunique()
                )
            
            with col2:
                st.metric(
                    label="Quarters Retrieved",
                    value=len(ms_quarters)
                )
            
            with col3:
                # Calculate QoQ gainer using latest available quarters in the data
                available_quarters = sorted([int(q.replace('Q', '')) for q in market_share_df['Quarter'].unique()])
                
                if len(available_quarters) >= 2:
                    # Use the two most recent quarters available in the data
                    last_quarter_num = max(available_quarters)
                    prev_quarter_num = sorted(available_quarters)[-2]  # Second highest
                    
                    last_quarter = f"Q{last_quarter_num}"
                    prev_quarter = f"Q{prev_quarter_num}"
                    
                    # Calculate QoQ change for each broker
                    qoq_data = []
                    for broker in market_share_df['Brokerage_Code'].unique():
                        broker_data = market_share_df[market_share_df['Brokerage_Code'] == broker]
                        last_q_data = broker_data[broker_data['Quarter'] == last_quarter]
                        prev_q_data = broker_data[broker_data['Quarter'] == prev_quarter]
                        
                        if not last_q_data.empty and not prev_q_data.empty:
                            last_share = last_q_data['Market_Share_Percent'].iloc[0]
                            prev_share = prev_q_data['Market_Share_Percent'].iloc[0]
                            qoq_change = last_share - prev_share
                            qoq_data.append({
                                'broker': broker,
                                'qoq_change': qoq_change,
                                'last_share': last_share,
                                'prev_share': prev_share
                            })
                    
                    if qoq_data:
                        top_gainer = max(qoq_data, key=lambda x: x['qoq_change'])
                        st.metric(
                            label=f"Top QoQ Gainer ({prev_quarter}→{last_quarter})",
                            value=f"{top_gainer['broker']}",
                            delta=f"+{top_gainer['qoq_change']:.2f}%"
                        )
                    else:
                        st.metric(label="Top QoQ Gainer", value="No data")
                else:
                    # Fallback to top broker if only one quarter available
                    if not market_share_df.empty:
                        top_broker = market_share_df.loc[market_share_df['Market_Share_Percent'].idxmax()]
                        st.metric(
                            label="Top Broker",
                            value=f"{top_broker['Brokerage_Code']}",
                            delta=f"{top_broker['Market_Share_Percent']:.2f}%"
                        )
                    else:
                        st.metric(label="Top Broker", value="No data")
            
            # Create pivot table for better display
            pivot_df = market_share_df.pivot_table(
                index=['Brokerage_Code', 'Brokerage_Name'],
                columns='Quarter',
                values='Market_Share_Percent',
                aggfunc='first'
            ).reset_index()
            
            # Fill NaN values with 0
            pivot_df = pivot_df.fillna(0)
            
            # Sort by largest to smallest using the latest available quarter in the data
            available_quarters = [col for col in pivot_df.columns if col.startswith('Q')]
            if available_quarters:
                # Get the latest quarter available in the data
                quarter_nums = [int(q.replace('Q', '')) for q in available_quarters]
                latest_quarter_num = max(quarter_nums)
                latest_quarter_col = f"Q{latest_quarter_num}"
                
                if latest_quarter_col in pivot_df.columns:
                    # Create a copy to avoid modifying the original
                    sort_df = pivot_df.copy()
                    
                    # Convert percentage strings back to numbers for sorting
                    sort_col = sort_df[latest_quarter_col].astype(str)
                    sort_col = sort_col.str.replace('%', '', regex=False)
                    sort_col = sort_col.str.replace('-', '0', regex=False)
                    sort_col = pd.to_numeric(sort_col, errors='coerce').fillna(0)
                    
                    # Sort by the numeric values (descending)
                    sort_idx = sort_col.argsort()[::-1]
                    pivot_df = pivot_df.iloc[sort_idx]
            
            # Format percentage columns
            for col in pivot_df.columns:
                if col.startswith('Q'):
                    pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.2f}%" if x > 0 else "-")
            
            # Display the table
            st.subheader("Market Share by Quarter")
            st.dataframe(
                pivot_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Create visualization
            if len(ms_quarters) > 1:
                st.subheader("Market Share Trends")
                
                # Get top 10 brokers by market share using the latest available quarter
                available_quarters = sorted([int(q.replace('Q', '')) for q in market_share_df['Quarter'].unique()])
                
                if available_quarters:
                    latest_quarter_num = max(available_quarters)
                    latest_quarter = f"Q{latest_quarter_num}"
                    latest_quarter_data = market_share_df[market_share_df['Quarter'] == latest_quarter]
                    
                    if not latest_quarter_data.empty:
                        top_10_brokers = latest_quarter_data.nlargest(10, 'Market_Share_Percent')['Brokerage_Code'].tolist()
                    else:
                        # Fallback to average if latest quarter has no data
                        avg_share = market_share_df.groupby('Brokerage_Code')['Market_Share_Percent'].mean().sort_values(ascending=False)
                        top_10_brokers = avg_share.head(10).index.tolist()
                else:
                    # Fallback to average market share
                    avg_share = market_share_df.groupby('Brokerage_Code')['Market_Share_Percent'].mean().sort_values(ascending=False)
                    top_10_brokers = avg_share.head(10).index.tolist()
                
                # Filter data for top 10 brokers
                top_brokers_data = market_share_df[market_share_df['Brokerage_Code'].isin(top_10_brokers)]
                
                # Create line chart
                fig = px.line(
                    top_brokers_data,
                    x='Quarter',
                    y='Market_Share_Percent',
                    color='Brokerage_Code',
                    title=f"Top 10 Brokers Market Share Trends - {ms_year}",
                    labels={
                        'Market_Share_Percent': 'Market Share (%)',
                        'Quarter': 'Quarter',
                        'Brokerage_Code': 'Broker'
                    }
                )
                
                fig.update_layout(
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Download option
            csv = market_share_df.to_csv(index=False)
            st.download_button(
                label="Download Market Share Data as CSV",
                data=csv,
                file_name=f"market_share_{ms_year}_Q{'_'.join(map(str, ms_quarters))}.csv",
                mime="text/csv"
            )
            
        else:
            st.warning("No market share data available for the selected parameters.")
    else:
        st.info("Please select at least one quarter in the Charts tab to display market share data.")