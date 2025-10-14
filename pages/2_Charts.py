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

    # If TOTAL_OPERATING_INCOME is requested, also load its components
    metrics_to_load = list(metrics)
    if 'TOTAL_OPERATING_INCOME' in metrics:
        # TOI = Fee Income + Capital Income (6 components total)
        toi_components = ['NET_BROKERAGE_INCOME', 'NET_IB_INCOME', 'NET_OTHER_OP_INCOME',
                         'NET_TRADING_INCOME', 'INTEREST_INCOME', 'MARGIN_LENDING_INCOME']
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

def get_metric_display_name(metric_code):
    """Convert metric code to display name"""
    metric_names = {
        'NET_BROKERAGE_INCOME': 'Brokerage Income',
        'NET_TRADING_INCOME': 'Trading Income',
        'INTEREST_INCOME': 'Interest Income',
        'NET_INVESTMENT_INCOME': 'Investment Income',
        'FEE_INCOME': 'Fee Income',
        'CAPITAL_INCOME': 'Capital Income',
        'MARGIN_BALANCE': 'Margin Balance',
        'TOTAL_OPERATING_INCOME': 'Total Operating Income',
        'NET_IB_INCOME': 'IB Income',
        'NET_OTHER_OP_INCOME': 'Other Income',
        'BORROWING_BALANCE': 'Borrowing Balance',
        'PBT': 'PBT',
        'NPAT': 'NPAT',
        'SGA': 'SG&A',
        'MARGIN_LENDING_INCOME': 'Margin Lending Income',
        'ROE': 'ROE',
        'ROA': 'ROA',
        'INTEREST_RATE': 'Interest Rate',
        'MARGIN_LENDING_RATE': 'Margin Lending Rate'
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
        'Brokerage Income': 'NET_BROKERAGE_INCOME',
        'Margin Lending Income': 'MARGIN_LENDING_INCOME',
        'Trading Income': 'NET_TRADING_INCOME',
        'Interest Income': 'INTEREST_INCOME',
        'IB Income': 'NET_IB_INCOME',
        'Other Operating Income': 'NET_OTHER_OP_INCOME'
    }

    # Collect data for each broker
    all_data = []

    for broker in selected_brokers:
        # Get TOI values
        toi_data = filtered_df[(filtered_df['TICKER'] == broker) &
                               (filtered_df['METRIC_CODE'] == 'TOTAL_OPERATING_INCOME')].copy()

        if toi_data.empty:
            continue

        toi_data = toi_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])

        for _, toi_row in toi_data.iterrows():
            year = toi_row['YEARREPORT']
            quarter = toi_row['LENGTHREPORT']
            quarter_label = toi_row.get('Quarter_Label', '')
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
    'NET_BROKERAGE_INCOME',
    'NET_TRADING_INCOME',
    'INTEREST_INCOME',
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
    'MARGIN_LENDING_INCOME',
    'ROE',
    'ROA',
    'INTEREST_RATE',
    'MARGIN_LENDING_RATE'
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
fixed_charts = ['PBT', 'ROE', 'TOTAL_OPERATING_INCOME']

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
                if metric == 'TOTAL_OPERATING_INCOME':
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
                                if metric in ['ROE', 'ROA', 'INTEREST_RATE', 'MARGIN_LENDING_RATE']:
                                    broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce') * 100  # Convert to percentage
                                    y_values = broker_data['DISPLAY_VALUE']
                                    hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.2f}}%<br><extra></extra>"

                                    if not broker_data_with_ma4.empty:
                                        broker_data_with_ma4['MA4_DISPLAY'] = broker_data_with_ma4['MA4'] * 100
                                else:
                                    # Convert other values to billions for display
                                    broker_data['DISPLAY_VALUE'] = pd.to_numeric(broker_data['VALUE'], errors='coerce') / 1_000_000_000
                                    y_values = broker_data['DISPLAY_VALUE']
                                    hover_template = f"<b>{broker}</b><br>Period: %{{x}}<br>Value: %{{y:,.3f}}B VND<br><extra></extra>"

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
                                            hovertemplate=f"<b>{broker} MA4</b><br>Period: %{{x}}<br>MA4: %{{y:,.2f}}<br><extra></extra>",
                                            legendgroup=broker
                                        )
                                    )

                        # Set y-axis title and format based on metric type
                        if metric in ['ROE', 'ROA', 'INTEREST_RATE', 'MARGIN_LENDING_RATE']:
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

                chart_idx += 1
        else:
            st.warning("Please select brokers to display charts.")
    else:
        st.info("Please select brokers and years to display charts")

with tab2:
    st.header("Brokerage Market Share Data")

    # Filters for Market Share tab
    st.subheader("Filters")

    col1, col2 = st.columns(2)

    with col1:
        # Year selection
        available_ms_years = list(range(2021, 2026))
        selected_ms_years = st.multiselect(
            "Select Years:",
            options=available_ms_years,
            default=[2024],
            key="market_share_years"
        )

    with col2:
        # Quarter selection
        selected_ms_quarters = st.multiselect(
            "Select Quarters:",
            options=[1, 2, 3, 4],
            default=[1, 2, 3, 4],
            format_func=lambda x: f"Q{x}",
            key="market_share_quarters"
        )

    # Broker selection for market share
    st.markdown("**Broker Selection:**")
    col3, col4 = st.columns([1, 4])

    with col3:
        show_all_brokers = st.checkbox(
            "Show All (Top 10)",
            value=True,
            key="show_all_market_share",
            help="Show all top 10 brokers by market share"
        )

    with col4:
        if not show_all_brokers:
            # Show broker multiselect only if "Show All" is unchecked
            selected_ms_brokers = st.multiselect(
                "Select Specific Brokers:",
                options=all_brokers_ordered,
                default=[],
                key="market_share_brokers",
                help="Select one or more brokers to display"
            )
        else:
            selected_ms_brokers = []  # Will be determined by top 10 in the data

    if selected_ms_years and selected_ms_quarters:
        # Fetch data for all selected year-quarter combinations
        with st.spinner("Fetching market share data..."):
            all_market_share_data = []
            for year in selected_ms_years:
                if 2021 <= year <= 2025:
                    year_data = fetch_market_share_data(year, selected_ms_quarters)
                    if not year_data.empty:
                        all_market_share_data.append(year_data)

            if all_market_share_data:
                market_share_df = pd.concat(all_market_share_data, ignore_index=True)
            else:
                market_share_df = pd.DataFrame()

        # Create period labels for display
        selected_quarter_labels = []
        for year in sorted(selected_ms_years):
            for quarter in sorted(selected_ms_quarters):
                selected_quarter_labels.append(f"Q{quarter} {year}")

        if not market_share_df.empty:
            # Create dynamic title based on selected quarters
            time_period_label = ", ".join(selected_quarter_labels)
            st.subheader(f"Market Share Data")
            
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
                    value=len(selected_quarter_labels)
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
            if len(selected_quarter_labels) > 1:
                st.subheader("Market Share Trends")

                # Determine which brokers to display
                if show_all_brokers:
                    # Get top 10 brokers by market share using the latest available quarter across all years
                    # Sort dataframe to get the most recent period
                    market_share_df_sorted = market_share_df.sort_values(['Year', 'Quarter'], ascending=[True, True])

                    # Get the latest period
                    latest_row = market_share_df_sorted.iloc[-1]
                    latest_quarter = latest_row['Quarter']
                    latest_year = latest_row['Year']

                    # Get data for the latest period
                    latest_quarter_data = market_share_df[
                        (market_share_df['Quarter'] == latest_quarter) &
                        (market_share_df['Year'] == latest_year)
                    ]

                    if not latest_quarter_data.empty:
                        brokers_to_display = latest_quarter_data.nlargest(10, 'Market_Share_Percent')['Brokerage_Code'].tolist()
                    else:
                        # Fallback to average if latest quarter has no data
                        avg_share = market_share_df.groupby('Brokerage_Code')['Market_Share_Percent'].mean().sort_values(ascending=False)
                        brokers_to_display = avg_share.head(10).index.tolist()
                else:
                    # Use user-selected brokers
                    if selected_ms_brokers:
                        brokers_to_display = selected_ms_brokers
                    else:
                        st.warning("Please select at least one broker or check 'Show All (Top 10)'")
                        brokers_to_display = []

                # Filter data for selected brokers
                if brokers_to_display:
                    top_brokers_data = market_share_df[market_share_df['Brokerage_Code'].isin(brokers_to_display)].copy()
                else:
                    top_brokers_data = pd.DataFrame()

                if not top_brokers_data.empty:
                    # Create a combined Period label for x-axis (e.g., "Q1 2024")
                    top_brokers_data['Period_Label'] = top_brokers_data['Quarter'] + ' ' + top_brokers_data['Year'].astype(str)

                    # Sort by Year and Quarter for proper ordering
                    top_brokers_data = top_brokers_data.sort_values(['Year', 'Quarter'])

                    # Determine chart title
                    if show_all_brokers:
                        chart_title = "Top 10 Brokers Market Share Trends"
                    else:
                        chart_title = f"Market Share Trends - {len(brokers_to_display)} Broker(s)"

                    # Create line chart
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
                        }
                    )
                
                    fig.update_layout(
                        height=500,
                        hovermode='x unified'
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    if not show_all_brokers and not selected_ms_brokers:
                        st.info("Please select at least one broker or check 'Show All (Top 10)' to view the chart.")
            
            # Download option
            csv = market_share_df.to_csv(index=False)
            # Create a filename based on selected quarters
            filename_periods = "_".join([label.replace(' ', '') for label in selected_quarter_labels[:3]])
            if len(selected_quarter_labels) > 3:
                filename_periods += f"_and_{len(selected_quarter_labels)-3}_more"
            st.download_button(
                label="Download Market Share Data as CSV",
                data=csv,
                file_name=f"market_share_{filename_periods}.csv",
                mime="text/csv"
            )

        else:
            st.warning("No market share data available for the selected parameters.")
    else:
        st.info("Select years and quarters above.")