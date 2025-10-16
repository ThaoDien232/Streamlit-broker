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

    # If calculated metrics are requested, load their base components
    metrics_to_load = list(metrics)

    # ALWAYS load TOI and its components since TOI structure is a fixed chart
    # TOI = Fee Income + Capital Income (6 components total)
    toi_components = ['Total_Operating_Income', 'Net_Brokerage_Income', 'Net_IB_Income', 'Net_other_operating_income',
                     'Net_Trading_Income', 'Net_Interest_Income', 'Net_Margin_lending_Income']
    metrics_to_load.extend([m for m in toi_components if m not in metrics_to_load])

    df = load_filtered_brokerage_data(
        tickers=tickers,
        metrics=metrics_to_load,
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


def get_available_market_share_periods(start_year: int = 2021) -> list[tuple[int, int]]:
    """Generate all available year-quarter combinations up to the current quarter."""
    periods: list[tuple[int, int]] = []
    today = pd.Timestamp.today()
    current_year = today.year
    current_quarter = (today.month - 1) // 3 + 1

    year = start_year
    while year < current_year:
        for quarter in range(1, 5):
            periods.append((year, quarter))
        year += 1

    for quarter in range(1, current_quarter + 1):
        periods.append((current_year, quarter))

    return periods


def get_recent_market_share_periods(count: int) -> list[tuple[int, int]]:
    """Return the most recent N year-quarter combinations."""
    periods = get_available_market_share_periods()
    if not periods:
        return []
    return periods[-count:] if count < len(periods) else periods

def get_metric_display_name(metric_code):
    """Convert metric code to display name"""
    metric_names = {
        'Net_Brokerage_Income': 'Brokerage Income',
        'Net_Trading_Income': 'Trading Income',
        'Net_Interest_Income': 'Interest Income',
        'Net_investment_income': 'Investment Income',
        'Net_Fee_Income': 'Fee Income',
        'Net_Capital_Income': 'Capital Income',
        'Margin_Lending_book': 'Margin Balance',
        'Total_Operating_Income': 'Total Operating Income',
        'Net_IB_Income': 'IB Income',
        'Net_other_operating_income': 'Other Income',
        'Borrowing_Balance': 'Borrowing Balance',
        'PBT': 'PBT',
        'NPAT': 'NPAT',
        'SGA': 'SG&A',
        'Net_Margin_lending_Income': 'Margin Lending Income',
        'ROE': 'ROE',
        'ROA': 'ROA',
        'INTEREST_RATE': 'Interest Rate',
        'MARGIN_LENDING_RATE': 'Margin Lending Rate',
        'MARGIN_LENDING_SPREAD': 'Margin Lending Spread',
        'NET_BROKERAGE_FEE': 'Net Brokerage Fee (bps)'
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

def calculate_ma4(df, metric_code, ticker):
    """Calculate MA4 (moving average of 4 most recent quarters) for a metric"""
    # Filter data for the specific metric and ticker
    metric_data = df[(df['METRIC_CODE'] == metric_code) & (df['TICKER'] == ticker)].copy()

    if metric_data.empty:
        return pd.DataFrame()

    # Sort by year and quarter
    metric_data = metric_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])

    # Calculate rolling average of 4 quarters
    metric_data['MA4'] = metric_data['VALUE'].rolling(window=4, min_periods=1).mean()

    return metric_data

def create_toi_structure_chart(filtered_df, selected_brokers, timeframe_type):
    """Create stacked bar chart showing TOI structure (income streams as % of TOI)"""

    # Validate inputs
    if filtered_df.empty or not selected_brokers:
        return None

    # TOI = Fee Income + Capital Income
    # Fee Income = Net Brokerage + Net IB + Net Other Operating
    # Capital Income = Net Trading + Interest + Margin Lending
    # So the 6 components that sum to 100% of TOI are:
    toi_components = {
        'Brokerage Income': 'Net_Brokerage_Income',
        'Margin Lending Income': 'Net_Margin_lending_Income',
        'Trading Income': 'Net_Trading_Income',
        'Interest Income': 'Net_Interest_Income',
        'IB Income': 'Net_IB_Income',
        'Other Operating Income': 'Net_other_operating_income'
    }

    # Collect data for each broker
    all_data = []

    for broker in selected_brokers:
        # Get TOI values
        toi_data = filtered_df[(filtered_df['TICKER'] == broker) &
                               (filtered_df['METRIC_CODE'] == 'Total_Operating_Income')].copy()

        if toi_data.empty:
            continue

        toi_data = toi_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])

        for _, toi_row in toi_data.iterrows():
            year = toi_row['YEARREPORT']
            quarter = toi_row['LENGTHREPORT']
            # Check both possible column names (Quarter_Label or QUARTER_LABEL)
            quarter_label = toi_row.get('Quarter_Label', toi_row.get('QUARTER_LABEL', ''))
            toi_value = toi_row['VALUE']

            # Skip if data is invalid
            if pd.isna(toi_value) or toi_value == 0 or not quarter_label:
                continue

            # Get component values for this period
            period_data = filtered_df[
                (filtered_df['TICKER'] == broker) &
                (filtered_df['YEARREPORT'] == year) &
                (filtered_df['LENGTHREPORT'] == quarter)
            ]

            for component_name, component_code in toi_components.items():
                component_row = period_data[period_data['METRIC_CODE'] == component_code]

                if not component_row.empty:
                    component_value = component_row.iloc[0]['VALUE']

                    # Skip if component_value is None or NaN
                    if pd.isna(component_value):
                        component_value = 0

                    percentage = (component_value / toi_value) * 100 if toi_value != 0 else 0

                    all_data.append({
                        'Broker': broker,
                        'Quarter_Label': quarter_label,
                        'Component': component_name,
                        'Percentage': percentage,
                        'Value': component_value,
                        'Year': year,
                        'Quarter': quarter
                    })

    if not all_data:
        return None

    df_structure = pd.DataFrame(all_data)

    # Create stacked bar chart for each broker
    fig = go.Figure()

    # Color scheme for components (6 components)
    component_colors = {
        'Brokerage Income': "#66c2a5",
        'IB Income': "#fc8d62",
        'Other Operating Income': "#cc96ff",
        'Trading Income': "#e5c494",
        'Interest Income': "#ffd92f",
        'Margin Lending Income': "#b0e0b0"
    }

    for broker in selected_brokers:
        broker_data = df_structure[df_structure['Broker'] == broker].copy()

        if broker_data.empty:
            continue

        # Sort by period
        broker_data = broker_data.sort_values(['Year', 'Quarter'])

        # Create subplots or separate traces for each broker
        for component_name in toi_components.keys():
            component_data = broker_data[broker_data['Component'] == component_name].copy()

            if not component_data.empty:
                # Validate data before adding trace
                x_values = component_data['Quarter_Label'].tolist()
                y_values = component_data['Percentage'].tolist()

                # Skip if data is invalid
                if not x_values or not y_values or len(x_values) != len(y_values):
                    continue

                # Convert to native Python types for Plotly
                x_values = [str(x) for x in x_values]  # Ensure strings
                y_values = [float(y) if not pd.isna(y) else 0.0 for y in y_values]  # Ensure floats

                # Skip if all values are zero
                if all(y == 0 for y in y_values):
                    continue

                try:
                    fig.add_trace(
                        go.Bar(
                            x=x_values,
                            y=y_values,
                            name=f"{broker} - {component_name}",
                            marker_color=component_colors.get(component_name, '#808080'),
                            hovertemplate=f"<b>{broker} - {component_name}</b><br>" +
                                        "Period: %{x}<br>" +
                                        "Percentage: %{y:.1f}%<br>" +
                                        "<extra></extra>",
                            legendgroup=broker
                        )
                    )
                except Exception as e:
                    st.warning(f"Skipping trace for {broker} - {component_name}: {str(e)}")

    fig.update_layout(
        barmode='stack',
        title=f"Total Operating Income Structure - {timeframe_type}",
        xaxis_title="Period",
        yaxis_title="Percentage of TOI (%)",
        height=500,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    # Set y-axis range to 0-100%
    fig.update_yaxes(range=[0, 100])

    return fig

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
    'Net_Brokerage_Income',
    'Net_Trading_Income',
    'Net_Interest_Income',
    'Net_investment_income',
    'Net_Fee_Income',
    'Net_Capital_Income',
    'Margin_Lending_book',
    'Total_Operating_Income',
    'Net_IB_Income',
    'Net_other_operating_income',
    'Borrowing_Balance',
    'PBT',  # Changed from full name
    'NPAT',  # Changed from full name
    'SGA',
    'Net_Margin_lending_Income',
    'ROE',
    'ROA',
    'INTEREST_RATE',
    'MARGIN_LENDING_RATE',
    'MARGIN_LENDING_SPREAD',
    'NET_BROKERAGE_FEE'
]

# Broker groups for organized display
broker_groups = {
    'Top Tier': ['SSI', 'VCI', 'VND', 'HCM', 'TCBS', 'VPBS', 'VPS'],
    'Mid Tier': ['MBS', 'VIX', 'SHS', 'BSI', 'FTS'],
    'Regional': ['DSE', 'VDS', 'LPBS', 'Kafi', 'ACBS', 'OCBS', 'HDBS'],
}

# Create grouped broker list maintaining order within groups
grouped_brokers = []
for group_name, group_brokers in broker_groups.items():
    # Add brokers from this group that exist in available_brokers
    for broker in group_brokers:
        if broker in available_brokers:
            grouped_brokers.append(broker)

# Add remaining brokers not in any group (sorted)
remaining_brokers = sorted([b for b in available_brokers if b not in grouped_brokers])
all_brokers_ordered = grouped_brokers + remaining_brokers

# Broker selection - NO DEFAULT
selected_brokers = st.sidebar.multiselect(
    "Select Brokers:",
    options=all_brokers_ordered,
    default=[],  # No default brokers
    key="chart_brokers",
    help="Brokers organized by tier: Top (SSI, VCI, VND, HCM, TCBS, VPBS, VPS) | Mid (MBS, VIX, SHS, BSI, FTS) | Regional (DSE, VDS, LPBS, Kafi, ACBS, OCBS, HDBS)"
)

# Fixed default charts (always displayed)
fixed_charts = ['Total_Operating_Income', 'PBT', 'ROE', 'MARGIN_LENDING_RATE', 'INTEREST_RATE', 'NET_BROKERAGE_FEE']

# Additional metrics selection - NOW ALWAYS AVAILABLE
additional_metrics = st.sidebar.multiselect(
    "Add charts:",
    options=[m for m in allowed_metrics if m not in fixed_charts],
    default=[],
    format_func=get_metric_display_name,
    key="chart_metrics"
)

# Combine fixed and additional metrics
selected_metrics = fixed_charts + additional_metrics

# Year selection - default to 2023, 2024, 2025
available_years_filtered = [year for year in available_years if 2021 <= year <= 2025]
default_years = [y for y in [2023, 2024, 2025] if y in available_years_filtered]
selected_years = st.sidebar.multiselect(
    "Select Years:",
    options=available_years_filtered if available_years_filtered else available_years,
    default=default_years,
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

            # Display charts - handle TOI structure specially
            chart_idx = 0
            for metric in selected_metrics:
                # Special handling for TOI Structure chart
                if metric == 'Total_Operating_Income':
                    st.subheader("Total Operating Income Structure")
                    toi_fig = create_toi_structure_chart(filtered_df, selected_brokers, timeframe_type)
                    if toi_fig:
                        st.plotly_chart(toi_fig, use_container_width=True)
                    else:
                        st.warning("No data available for TOI structure chart.")
                    continue

                # Standard charts (PBT, ROE, and additional metrics)
                # Display 2 charts per row
                if chart_idx % 2 == 0:
                    col1, col2 = st.columns(2)

                with col1 if chart_idx % 2 == 0 else col2:
                    st.subheader(f"{get_metric_display_name(metric)}")

                    # Filter data for current metric
                    metric_data = filtered_df[filtered_df['METRIC_CODE'] == metric].copy()

                    if not metric_data.empty:
                        # Sort data chronologically
                        metric_data = metric_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])

                        # Create bar chart
                        fig = go.Figure()

                        # Color palette
                        colors = px.colors.qualitative.Set2

                        # Add bar and MA4 line for each broker
                        for j, broker in enumerate(selected_brokers):
                            broker_data = metric_data[metric_data['TICKER'] == broker].copy()

                            if not broker_data.empty:
                                # Sort by period to ensure consistent ordering
                                broker_data = broker_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])

                                # Calculate MA4 for this broker
                                broker_data_with_ma4 = calculate_ma4(filtered_df, metric, broker)

                                # Check if this is ROE, ROA, or rate metrics (percentage metrics)
                                if metric in ['INTEREST_RATE', 'MARGIN_LENDING_RATE', 'MARGIN_LENDING_SPREAD']:
                                    broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce')  # Already in percentage
                                    y_values = broker_data['DISPLAY_VALUE']
                                    hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.2f}}%<br><extra></extra>"
                                    ma4_hover_template = f"<b>{broker} MA4</b><br>Period: %{{x}}<br>MA4: %{{y:,.2f}}%<br><extra></extra>"

                                    if not broker_data_with_ma4.empty:
                                        broker_data_with_ma4['MA4_DISPLAY'] = broker_data_with_ma4['MA4']  # Already in percentage

                                elif  metric in ['ROE', 'ROA']:
                                    broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce')  # Already in percentage
                                    y_values = broker_data['DISPLAY_VALUE']
                                    hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.2f}}%<br><extra></extra>"
                                    ma4_hover_template = f"<b>{broker} MA4</b><br>Period: %{{x}}<br>MA4: %{{y:,.2f}}%<br><extra></extra>"

                                    if not broker_data_with_ma4.empty:
                                        broker_data_with_ma4['MA4_DISPLAY'] = broker_data_with_ma4['MA4']  # Already in percentage
                                elif metric == 'NET_BROKERAGE_FEE':
                                    # Net Brokerage Fee is already in basis points
                                    broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce')
                                    y_values = broker_data['DISPLAY_VALUE']
                                    hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.2f}} bps<br><extra></extra>"
                                    ma4_hover_template = f"<b>{broker} MA4</b><br>Period: %{{x}}<br>MA4: %{{y:,.2f}} bps<br><extra></extra>"

                                    if not broker_data_with_ma4.empty:
                                        broker_data_with_ma4['MA4_DISPLAY'] = broker_data_with_ma4['MA4']
                                else:
                                    # Convert other values to billions for display
                                    broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce') / 1_000_000_000
                                    y_values = broker_data['DISPLAY_VALUE']
                                    hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.3f}}B VND<br><extra></extra>"
                                    ma4_hover_template = f"<b>{broker} MA4</b><br>Period: %{{x}}<br>MA4: %{{y:,.3f}}B VND<br><extra></extra>"

                                    if not broker_data_with_ma4.empty:
                                        broker_data_with_ma4['MA4_DISPLAY'] = broker_data_with_ma4['MA4'] / 1_000_000_000

                                # Add bar trace
                                fig.add_trace(
                                    go.Bar(
                                        x=broker_data['Quarter_Label'],
                                        y=y_values,
                                        name=broker,
                                        marker_color=colors[j % len(colors)],
                                        hovertemplate=hover_template,
                                        legendgroup=broker
                                    )
                                )

                                # Add MA4 line trace - solid line with prominent color
                                if not broker_data_with_ma4.empty and len(broker_data_with_ma4) >= 4:
                                    # Use contrasting colors for MA4 lines to make them stand out
                                    ma4_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFD700']
                                    fig.add_trace(
                                        go.Scatter(
                                            x=broker_data_with_ma4['Quarter_Label'],
                                            y=broker_data_with_ma4['MA4_DISPLAY'],
                                            name=f"{broker} MA4",
                                            mode='lines+markers',
                                            line=dict(color=ma4_colors[j % len(ma4_colors)], width=3),
                                            marker=dict(size=8, symbol='diamond'),
                                            hovertemplate=ma4_hover_template,
                                            legendgroup=broker
                                        )
                                    )

                        # Set y-axis title and format based on metric type
                        if metric in ['ROE', 'ROA', 'INTEREST_RATE', 'MARGIN_LENDING_RATE', 'MARGIN_LENDING_SPREAD']:
                            yaxis_title = "Percentage (%)"
                            tick_format = ".2f"
                        elif metric == 'NET_BROKERAGE_FEE':
                            yaxis_title = "Basis Points (bps)"
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

                chart_idx += 1
        else:
            st.warning("Please select brokers to display charts.")
    else:
        st.info("Please select brokers and years to display charts")

with tab2:
    st.header("Brokerage Market Share Data")

    # Filters for Market Share tab
    st.subheader("Filters")

    col_recent, col_info = st.columns([1, 1])

    with col_recent:
        recent_quarter_count = int(
            st.number_input(
                "Most recent quarters",
                min_value=1,
                max_value=16,
                value=6,
                step=1,
                key="market_share_recent_periods"
            )
        )

    with col_info:
        st.caption("View aggregated HSX top-10 brokerage market share for the latest N quarters.")

    selected_periods = get_recent_market_share_periods(recent_quarter_count)
    if not selected_periods:
        st.warning("No market share periods available. Please try again later.")
        st.stop()

    # Determine top 10 brokers based on the most recent period
    latest_year, latest_quarter = selected_periods[-1]
    sample_data = fetch_market_share_data(latest_year, [latest_quarter])

    if not sample_data.empty:
        top_10_broker_codes = sample_data.nlargest(10, 'Market_Share_Percent')['Brokerage_Code'].tolist()
        top_10_options = top_10_broker_codes + ['All']
    else:
        top_10_options = ['All']

    # Broker selection for market share
    selected_ms_brokers = st.multiselect(
        "Select Brokers (from Top 10):",
        options=top_10_options,
        default=['All'],
        key="market_share_brokers",
        help="Select one or more brokers from top 10, or 'All' to show all top 10"
    )

    if selected_periods:
        # Fetch data for all selected periods
        with st.spinner("Fetching market share data..."):
            all_market_share_data = []
            for year, quarter in selected_periods:
                year_data = fetch_market_share_data(year, [quarter])
                if not year_data.empty:
                    all_market_share_data.append(year_data)

            if all_market_share_data:
                market_share_df = pd.concat(all_market_share_data, ignore_index=True)
            else:
                market_share_df = pd.DataFrame()

        # Create period labels for display
        selected_quarter_labels = [f"{year} Q{quarter}" for year, quarter in selected_periods]

        if not market_share_df.empty:
            market_share_df['Market_Share_Percent'] = pd.to_numeric(market_share_df['Market_Share_Percent'], errors='coerce')
            market_share_df = market_share_df.dropna(subset=['Market_Share_Percent'])

            market_share_df['Quarter_Number'] = pd.to_numeric(
                market_share_df['Quarter'].astype(str).str.replace('Q', ''),
                errors='coerce'
            )
            market_share_df = market_share_df.dropna(subset=['Quarter_Number'])
            market_share_df['Quarter_Number'] = market_share_df['Quarter_Number'].astype(int)
            market_share_df['Period_Label'] = market_share_df.apply(
                lambda row: f"{int(row['Year'])} Q{int(row['Quarter_Number'])}", axis=1
            )
            period_order_map = {
                (year, quarter): idx for idx, (year, quarter) in enumerate(selected_periods)
            }
            market_share_df['Period_Order'] = market_share_df.apply(
                lambda row: period_order_map.get((int(row['Year']), int(row['Quarter_Number']))),
                axis=1
            )

            market_share_df = market_share_df[market_share_df['Period_Order'].notna()].copy()
            market_share_df = market_share_df.sort_values(['Period_Order', 'Brokerage_Code']).reset_index(drop=True)

            if market_share_df.empty:
                st.warning("No market share data available for the selected quarters.")
            else:
                st.subheader("Market Share Data")

                period_labels = [f"{year} Q{quarter}" for year, quarter in selected_periods]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="Total Brokers",
                        value=market_share_df['Brokerage_Code'].nunique()
                    )

                with col2:
                    st.metric(
                        label="Quarters Retrieved",
                        value=len(period_labels)
                    )

                with col3:
                    if len(selected_periods) >= 2:
                        prev_year, prev_quarter = selected_periods[-2]
                        last_year, last_quarter = selected_periods[-1]
                        prev_label = f"{prev_year} Q{prev_quarter}"
                        last_label = f"{last_year} Q{last_quarter}"

                        last_data = market_share_df[
                            (market_share_df['Year'] == last_year)
                            & (market_share_df['Quarter_Number'] == last_quarter)
                        ]
                        prev_data = market_share_df[
                            (market_share_df['Year'] == prev_year)
                            & (market_share_df['Quarter_Number'] == prev_quarter)
                        ]

                        if not last_data.empty and not prev_data.empty:
                            last_series = last_data.set_index('Brokerage_Code')['Market_Share_Percent']
                            prev_series = prev_data.set_index('Brokerage_Code')['Market_Share_Percent']
                            common_brokers = last_series.index.intersection(prev_series.index)

                            qoq_data = []
                            for broker in common_brokers:
                                last_share = float(last_series.loc[broker])
                                prev_share = float(prev_series.loc[broker])
                                qoq_data.append({
                                    'broker': broker,
                                    'qoq_change': last_share - prev_share,
                                    'last_share': last_share,
                                    'prev_share': prev_share
                                })

                            if qoq_data:
                                top_gainer = max(qoq_data, key=lambda x: x['qoq_change'])
                                st.metric(
                                    label=f"Top QoQ Gainer ({prev_label}→{last_label})",
                                    value=top_gainer['broker'],
                                    delta=f"{top_gainer['qoq_change']:+.2f}%"
                                )
                            else:
                                st.metric(label="Top QoQ Gainer", value="No data")
                        else:
                            st.metric(label="Top QoQ Gainer", value="No data")
                    else:
                        top_row = market_share_df.loc[market_share_df['Market_Share_Percent'].idxmax()]
                        st.metric(
                            label="Top Broker",
                            value=top_row['Brokerage_Code'],
                            delta=f"{top_row['Market_Share_Percent']:.2f}%"
                        )

                # Pivot table display
                pivot_df = market_share_df.pivot_table(
                    index=['Brokerage_Code', 'Brokerage_Name'],
                    columns='Period_Label',
                    values='Market_Share_Percent',
                    aggfunc='first'
                ).reset_index()

                period_columns = [label for label in period_labels if label in pivot_df.columns]
                pivot_df = pivot_df[['Brokerage_Code', 'Brokerage_Name', *period_columns]].fillna(0)

                if period_columns:
                    latest_label = period_columns[-1]
                    pivot_df = pivot_df.sort_values(by=latest_label, ascending=False)

                for col in period_columns:
                    pivot_df[col] = pivot_df[col].apply(
                        lambda x: f"{x:.2f}%" if pd.notna(x) and x > 0 else "-"
                    )

                st.subheader("Market Share by Quarter")
                st.dataframe(
                    pivot_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Visualization
                if len(period_columns) > 1:
                    st.subheader("Market Share Trends")

                    if 'All' in selected_ms_brokers or not selected_ms_brokers:
                        latest_year, latest_quarter = selected_periods[-1]
                        latest_quarter_data = market_share_df[
                            (market_share_df['Year'] == latest_year)
                            & (market_share_df['Quarter_Number'] == latest_quarter)
                        ]

                        if not latest_quarter_data.empty:
                            brokers_to_display = latest_quarter_data.nlargest(10, 'Market_Share_Percent')['Brokerage_Code'].tolist()
                        else:
                            avg_share = market_share_df.groupby('Brokerage_Code')['Market_Share_Percent'].mean().sort_values(ascending=False)
                            brokers_to_display = avg_share.head(10).index.tolist()
                    else:
                        brokers_to_display = [b for b in selected_ms_brokers if b != 'All']

                    if brokers_to_display:
                        top_brokers_data = market_share_df[market_share_df['Brokerage_Code'].isin(brokers_to_display)].copy()
                    else:
                        top_brokers_data = pd.DataFrame()

                    if not top_brokers_data.empty:
                        top_brokers_data = top_brokers_data.sort_values('Period_Order')

                        chart_title = "Top 10 Brokers Market Share Trends" if ('All' in selected_ms_brokers or not selected_ms_brokers) else f"Market Share Trends - {len(brokers_to_display)} Broker(s)"

                        fig = px.line(
                            top_brokers_data,
                            x='Period_Label',
                            y='Market_Share_Percent',
                            color='Brokerage_Code',
                            title=chart_title,
                            labels={
                                'Market_Share_Percent': 'Market Share (%)',
                                'Period_Label': 'Period',
                                'Brokerage_Code': 'Broker'
                            },
                            category_orders={'Period_Label': period_columns}
                        )

                        fig.update_layout(height=500, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)

                csv = market_share_df.to_csv(index=False)
                filename_periods = "_".join([label.replace(' ', '') for label in period_labels[:3]])
                if len(period_labels) > 3:
                    filename_periods += f"_and_{len(period_labels)-3}_more"

                st.download_button(
                    label="Download Market Share Data as CSV",
                    data=csv,
                    file_name=f"market_share_{filename_periods}.csv",
                    mime="text/csv"
                )

        else:
            st.warning("No market share data available for the selected quarters.")
    else:
        st.info("Adjust the number of recent quarters above.")
