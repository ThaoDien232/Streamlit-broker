"""
Brokerage Sector Valuation Analysis Page
Provides comprehensive valuation metrics analysis for Vietnamese brokerage sector
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Valuation Analysis",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import toml
from datetime import timedelta

# Load theme from config.toml
theme_config = toml.load("utils/config.toml")
theme = theme_config["theme"]
primary_color = theme["primaryColor"]
background_color = theme["backgroundColor"]
secondary_background_color = theme["secondaryBackgroundColor"]
text_color = theme["textColor"]
font_family = theme["font"]

# Import utilities
from utils.valuation_analysis import (
    get_metric_column,
    calculate_historical_stats,
    prepare_statistics_table,
    get_sector_and_components,
    get_valuation_status,
    generate_valuation_histogram,
    create_sector_aggregates
)
from utils.valuation_data import load_brokerage_valuation_data

@st.cache_data(ttl=1800)
def load_valuation_data() -> pd.DataFrame:
    """Load valuation data with caching"""
    df = load_brokerage_valuation_data()
    if df.empty:
        return df
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
    return df

# Title and description
st.title("Valuation Analysis")
st.markdown("Comprehensive valuation metrics analysis with distribution charts, historical trends, and statistical measures")

# Manual refresh control
with st.sidebar:
    st.header("🔄 Data Controls")
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # Include Unlisted Brokers toggle
    include_unlisted = st.checkbox(
        "Include Unlisted Brokers",
        value=True,
        help="Toggle to include/exclude unlisted brokers from analysis"
    )

# Load data
df = load_valuation_data()

if df.empty:
    st.error("❌ Valuation data not found. Unable to connect to SQL database or no brokerage data available.")
    st.info("📝 **Required data**: PE, PB, PS, EV_EBITDA metrics from dbo.Market_Data table for brokerage tickers")
    st.info("🔍 **Check**: Database connection in streamlit secrets and broker ticker availability in Market_Data table")
    st.stop()

# Build sector aggregates
df = create_sector_aggregates(df)

# Filter data based on unlisted inclusion
if not include_unlisted:
    df = df[df['Type'] != 'Unlisted']

# Sidebar metric selection
with st.sidebar:
    st.markdown("### 📊 Analysis Settings")

    # Get available metrics
    available_metrics = [col for col in df.columns
                        if col not in ['TICKER', 'TRADE_DATE', 'Type', 'YEARREPORT', 'LENGTHREPORT']]

    if not available_metrics:
        st.error("No valuation metrics found in data")
        st.stop()

    # Metric selection - ONLY P/E and P/B valuation metrics
    metric_display_names = {
        'PE': 'Price-to-Earnings (P/E)',
        'PB': 'Price-to-Book (P/B)'
    }

    # Create options for selectbox - only include PE and PB if available
    metric_options = []
    for metric in ['PE', 'PB']:
        if metric in available_metrics:
            display_name = metric_display_names[metric]
            metric_options.append(display_name)

    selected_metric_display = st.selectbox(
        "Valuation Metric:",
        metric_options,
        index=0,
        help="This selection will update all charts and tables"
    )

    # Get the actual column name
    metric_type = None
    for metric, display_name in metric_display_names.items():
        if display_name == selected_metric_display:
            metric_type = metric
            break

    if metric_type is None:
        metric_type = available_metrics[0]

    metric_col = get_metric_column(metric_type)

# Get latest date in data
latest_date = df['TRADE_DATE'].max()

# Show latest data in smaller text
st.caption(f"Latest data: {latest_date.strftime('%Y-%m-%d')} | Metric: {selected_metric_display}")

# Chart 1: Valuation Distribution Candle Chart
st.subheader("📈 Valuation Distribution by Broker")

# Sector selection above the chart
sector_options = ["All_Brokers", "Listed", "Unlisted"] if include_unlisted else ["All_Brokers", "Listed"]
selected_sector = st.selectbox(
    "Select Sector:",
    sector_options,
    help="Shows selected sector plus all component brokers"
)

# Get tickers to display
display_tickers = get_sector_and_components(df, selected_sector, include_unlisted)

# Create candle chart
fig_candle = go.Figure()

# Prepare data for each ticker
valid_tickers = []
for ticker in display_tickers:
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()

    if len(ticker_data) < 5:  # Skip if insufficient data
        continue

    valid_tickers.append(ticker)

    # Calculate percentiles with smart outlier handling
    median_val = ticker_data.median()

    # Remove extreme outliers based on metric type
    if metric_type in ['ROE', 'ROA']:
        # For percentages, cap at reasonable ranges
        upper_limit = min(1.0, median_val * 3) if median_val > 0 else 1.0  # 100% max
        lower_limit = max(-0.5, median_val - 0.5) if median_val > 0 else -0.5  # -50% min
        clean_data = ticker_data[(ticker_data >= lower_limit) & (ticker_data <= upper_limit)]
    else:
        # For absolute values, use IQR-based filtering
        Q1 = ticker_data.quantile(0.25)
        Q3 = ticker_data.quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 2 * IQR
        lower_limit = Q1 - 2 * IQR
        clean_data = ticker_data[(ticker_data >= lower_limit) & (ticker_data <= upper_limit)]

    # Ensure we still have enough data
    if len(clean_data) < 3:
        clean_data = ticker_data

    # Calculate percentiles for candle
    p5 = clean_data.quantile(0.05)
    p25 = clean_data.quantile(0.25)
    p50 = clean_data.quantile(0.50)
    p75 = clean_data.quantile(0.75)
    p95 = clean_data.quantile(0.95)

    # Get current value
    current_val = ticker_data.iloc[-1] if len(ticker_data) > 0 else None

    # Format values based on metric type
    def format_value(val):
        if metric_type in ['ROE', 'ROA']:
            return f"{val*100:.1f}%" if pd.notna(val) else "N/A"
        elif metric_type in ['NPAT', 'TOTAL_EQUITY', 'TOTAL_ASSETS']:
            return f"{val/1e9:.1f}B" if pd.notna(val) and val >= 1e9 else f"{val/1e6:.0f}M" if pd.notna(val) else "N/A"
        else:
            return f"{val:.2f}" if pd.notna(val) else "N/A"

    # Add candlestick with light grey color
    fig_candle.add_trace(go.Candlestick(
        x=[ticker],
        open=[p25],
        high=[p95],
        low=[p5],
        close=[p75],
        name=ticker,
        showlegend=False,
        increasing_line_color='lightgrey',
        decreasing_line_color='lightgrey',
        hovertext=f"{ticker}<br>Median: {format_value(p50)}"
    ))

    # Add current value as scatter point
    if current_val and not pd.isna(current_val):
        # Calculate percentile
        percentile = np.sum(clean_data <= current_val) / len(clean_data) * 100

        fig_candle.add_trace(go.Scatter(
            x=[ticker],
            y=[current_val],
            mode='markers',
            marker=dict(size=8, color=primary_color, symbol='circle'),
            name=f"{ticker} Current",
            showlegend=False,
            hovertemplate=(
                f"<b>{ticker}</b><br>" +
                f"Current: {format_value(current_val)}<br>" +
                f"Percentile: {percentile:.1f}%<br>" +
                f"Median: {format_value(p50)}<br>" +
                "<extra></extra>"
            )
        ))

# Update layout
y_title = selected_metric_display
if metric_type in ['ROE', 'ROA']:
    y_title += " (%)"
elif metric_type in ['NPAT', 'TOTAL_EQUITY', 'TOTAL_ASSETS']:
    y_title += " (VND)"

fig_candle.update_layout(
    title=f"{selected_metric_display} Distribution - {selected_sector}",
    xaxis_title="Broker",
    yaxis_title=y_title,
    height=500,
    hovermode='x unified',
    xaxis=dict(
        categoryorder='array',
        categoryarray=valid_tickers,
        rangeslider=dict(visible=False),
        fixedrange=True
    ),
    yaxis=dict(fixedrange=True),
    dragmode=False
)

st.plotly_chart(fig_candle, use_container_width=True, config={'displayModeBar': False})

# Combined Ticker Selection for Charts 2 and 3
st.markdown("---")
st.subheader("🔍 Individual Broker Analysis")
st.caption("Select a broker to view both historical trend and distribution analysis")

# Common ticker and date range selection
col_select1, col_select2, col_select3 = st.columns([2, 2, 6])

with col_select1:
    # Ticker selection for both charts
    all_tickers = sorted([t for t in df['TICKER'].unique() if t not in ['Sector']])
    if not include_unlisted:
        all_tickers = [t for t in all_tickers if len(str(t)) == 3 and str(t).isalpha() or t in ['All_Brokers', 'Listed']]

    selected_ticker = st.selectbox(
        "Select Broker:",
        all_tickers,
        index=all_tickers.index('All_Brokers') if 'All_Brokers' in all_tickers else 0,
        key="common_ticker_select"
    )

with col_select2:
    # Date range selection for time series
    date_range = st.selectbox(
        "Time Period:",
        ["1 Year", "2 Years", "3 Years", "5 Years", "All Time"],
        index=2  # Default to 3 years
    )

    # Calculate date filter
    if date_range == "1 Year":
        start_date = latest_date - timedelta(days=365)
    elif date_range == "2 Years":
        start_date = latest_date - timedelta(days=730)
    elif date_range == "3 Years":
        start_date = latest_date - timedelta(days=1095)
    elif date_range == "5 Years":
        start_date = latest_date - timedelta(days=1825)
    else:
        start_date = df['TRADE_DATE'].min()

# Display charts side by side
col_chart1, col_chart2 = st.columns([6, 6])

# Chart 2: Historical Valuation Time Series
with col_chart1:
    # Filter data for selected ticker and date range
    ticker_df = df[(df['TICKER'] == selected_ticker) & (df['TRADE_DATE'] >= start_date)].copy()
    ticker_df = ticker_df.sort_values('TRADE_DATE')

    if len(ticker_df) > 0:
        # Calculate statistics
        hist_stats = calculate_historical_stats(df, selected_ticker, metric_col)

        if hist_stats:
            # Create figure
            fig_ts = go.Figure()

            # Add main valuation line
            fig_ts.add_trace(go.Scatter(
                x=ticker_df['TRADE_DATE'],
                y=ticker_df[metric_col],
                mode='lines+markers',
                name=f'{selected_metric_display}',
                line=dict(color=primary_color, width=2),
                marker=dict(size=4)
            ))

            # Add mean line
            fig_ts.add_trace(go.Scatter(
                x=[ticker_df['TRADE_DATE'].min(), ticker_df['TRADE_DATE'].max()],
                y=[hist_stats['mean'], hist_stats['mean']],
                mode='lines',
                name='Mean',
                line=dict(color='black', width=2, dash='solid')
            ))

            # Add +1 SD line
            fig_ts.add_trace(go.Scatter(
                x=[ticker_df['TRADE_DATE'].min(), ticker_df['TRADE_DATE'].max()],
                y=[hist_stats['upper_1sd'], hist_stats['upper_1sd']],
                mode='lines',
                name='+1 SD',
                line=dict(color='red', width=1, dash='dash')
            ))

            # Add -1 SD line
            fig_ts.add_trace(go.Scatter(
                x=[ticker_df['TRADE_DATE'].min(), ticker_df['TRADE_DATE'].max()],
                y=[hist_stats['lower_1sd'], hist_stats['lower_1sd']],
                mode='lines',
                name='-1 SD',
                line=dict(color='green', width=1, dash='dash')
            ))

            # Update layout
            fig_ts.update_layout(
                title=f"{selected_ticker} - {selected_metric_display} Trend",
                xaxis_title="Date",
                yaxis_title=y_title,
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_ts, use_container_width=True)

            # Show statistics below the chart
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                def format_metric_value(val):
                    if val is None or pd.isna(val):
                        return "N/A"
                    if metric_type in ['ROE', 'ROA']:
                        return f"{val*100:.2f}%"
                    elif metric_type in ['NPAT', 'TOTAL_EQUITY', 'TOTAL_ASSETS']:
                        return f"{val/1e9:.2f}B VND" if val >= 1e9 else f"{val/1e6:.0f}M VND"
                    else:
                        return f"{val:.2f}"

                st.metric("Current", format_metric_value(hist_stats['current']))
                st.metric("Mean", format_metric_value(hist_stats['mean']))
            with col_stat2:
                st.metric("Std Dev", format_metric_value(hist_stats['std']))
                z_score = hist_stats.get('z_score')
                if z_score is not None:
                    status, color = get_valuation_status(z_score, metric_type)
                    st.metric("Z-Score", f"{z_score:.2f}", delta=status)
        else:
            st.warning(f"Insufficient data for statistical analysis of {selected_ticker}")
    else:
        st.warning(f"No data available for {selected_ticker} in the selected time period")

# Chart 3: Valuation Distribution Histogram
with col_chart2:
    # Generate histogram data for the same selected ticker
    hist_data = generate_valuation_histogram(df, selected_ticker, metric_col)

    if hist_data:
        # Create histogram figure
        fig_hist = go.Figure()

        # Create bar colors - highlight current bin
        bar_colors = ['#E0E0E0'] * len(hist_data['counts'])
        if hist_data['current_bin_idx'] is not None:
            bar_colors[hist_data['current_bin_idx']] = primary_color

        # Add bars
        fig_hist.add_trace(go.Bar(
            x=hist_data['bin_labels'],
            y=hist_data['counts'],
            marker_color=bar_colors,
            text=hist_data['counts'],
            textposition='auto',
            showlegend=False,
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))

        # Format current value for display
        def format_hist_value(val):
            if metric_type in ['ROE', 'ROA']:
                return f"{val*100:.1f}%"
            elif metric_type in ['NPAT', 'TOTAL_EQUITY', 'TOTAL_ASSETS']:
                return f"{val/1e9:.1f}B" if val >= 1e9 else f"{val/1e6:.0f}M"
            else:
                return f"{val:.2f}"

        # Update layout
        fig_hist.update_layout(
            title=dict(
                text=f"{selected_ticker} - {selected_metric_display} Distribution<br>" +
                     f"<sub>Current: {format_hist_value(hist_data['current_value'])} (CDF: {hist_data['percentile']:.1f}%)</sub>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=f"{selected_metric_display} Range",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
            hovermode='x',
            bargap=0.1
        )

        # Display histogram
        st.plotly_chart(fig_hist, use_container_width=True)

        # Show distribution statistics below the histogram
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.metric("Current Value", format_hist_value(hist_data['current_value']))
            st.metric("Percentile (CDF)", f"{hist_data['percentile']:.1f}%")
        with col_dist2:
            st.metric("Median", format_hist_value(hist_data['median']))
            st.metric("Data Points", hist_data['n_total'])
    else:
        st.info(f"Insufficient data to generate histogram for {selected_ticker}")

# Table: Valuation Statistics Table
st.markdown("---")
st.subheader("📊 Valuation Statistics Summary")

# Prepare statistics table
stats_df = prepare_statistics_table(df, metric_col, metric_type)

if not stats_df.empty:
    # Create sortable table using st.dataframe with built-in sorting
    table_df = stats_df.copy()

    # Remove Type column for display
    if 'Type' in table_df.columns:
        table_df = table_df.drop('Type', axis=1)

    # Format numeric columns for better display
    def format_table_value(val, col_name):
        if pd.isna(val):
            return None
        if col_name in ['Current', 'Mean']:
            if metric_type in ['ROE', 'ROA']:
                return f"{val*100:.2f}%"
            elif metric_type in ['NPAT', 'TOTAL_EQUITY', 'TOTAL_ASSETS']:
                return f"{val/1e9:.1f}B" if val >= 1e9 else f"{val/1e6:.0f}M"
            else:
                return round(val, 2)
        elif col_name == 'CDF (%)':
            return f"{val:.1f}%"
        elif col_name == 'Z-Score':
            return round(val, 2)
        else:
            return val

    # Format the dataframe for display
    display_df = table_df.copy()
    for col in ['Current', 'Mean', 'CDF (%)', 'Z-Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: format_table_value(x, col))

    # Create a function to assign colors based on status
    def get_status_color(status):
        if status in ["Very Cheap", "Excellent"]:
            return "#90EE90"  # Light green
        elif status in ["Cheap", "Good"]:
            return "#B8E6B8"  # Lighter green
        elif status in ["Fair", "Average"]:
            return "#FFFFCC"  # Light yellow
        elif status in ["Expensive", "Below Average"]:
            return "#FFD4A3"  # Light orange
        elif status in ["Very Expensive", "Poor"]:
            return "#FFB3B3"  # Light red
        else:
            return None

    # Apply styling with color-coding
    def style_status(row):
        color = get_status_color(row['Status'])
        if color:
            return [f'background-color: {color}'] * len(row)
        else:
            return [''] * len(row)

    # Apply the styling
    styled_df = display_df.style.apply(style_status, axis=1)

    # Display sortable dataframe with color coding
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(600, len(display_df) * 35 + 100),
        column_config={
            "Ticker": st.column_config.TextColumn(
                "Ticker",
                help="Click column header to sort",
                width="small"
            ),
            "Current": st.column_config.TextColumn(
                "Current",
                help="Current valuation metric"
            ),
            "Mean": st.column_config.TextColumn(
                "Mean",
                help="Historical average"
            ),
            "CDF (%)": st.column_config.TextColumn(
                "CDF (%)",
                help="Percentile ranking"
            ),
            "Z-Score": st.column_config.TextColumn(
                "Z-Score",
                help="Standard deviations from mean"
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                help="Valuation assessment"
            )
        }
    )


    # Summary statistics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    # Count by status categories (adapt for different metric types)
    if metric_type in ['ROE', 'ROA', 'NPAT']:
        with col1:
            excellent_count = len(stats_df[stats_df['Status'].isin(['Excellent', 'Very Cheap'])])
            st.metric("Excellent Performance", excellent_count)

        with col2:
            good_count = len(stats_df[stats_df['Status'].isin(['Good', 'Cheap'])])
            st.metric("Good Performance", good_count)

        with col3:
            average_count = len(stats_df[stats_df['Status'].isin(['Average', 'Fair'])])
            st.metric("Average Performance", average_count)

        with col4:
            poor_count = len(stats_df[stats_df['Status'].isin(['Below Average', 'Poor', 'Expensive', 'Very Expensive'])])
            st.metric("Below Average", poor_count)
    else:
        with col1:
            high_count = len(stats_df[stats_df['Status'].isin(['Very High', 'Excellent'])])
            st.metric("High Values", high_count)

        with col2:
            average_count = len(stats_df[stats_df['Status'] == 'Average'])
            st.metric("Average Values", average_count)

        with col3:
            low_count = len(stats_df[stats_df['Status'].isin(['Low', 'Very Low'])])
            st.metric("Low Values", low_count)

        with col4:
            total_count = len(stats_df)
            st.metric("Total Brokers", total_count)

else:
    st.warning("Insufficient data to generate statistics table")

# Data Quality Information
with st.expander("ℹ️ Data Information"):
    st.markdown(f"""
    **Data Source**: Combined Financial Data (calculated metrics)

    **Available Metrics**: {', '.join(available_metrics)}

    **Broker Classification**:
    - **Listed Brokers**: 3-letter ticker symbols (e.g., SSI, VCI, HCM)
    - **Unlisted Brokers**: Non-3-letter ticker symbols

    **Statistical Analysis**:
    - **Z-Score**: Number of standard deviations from historical mean
    - **CDF**: Cumulative Distribution Function (percentile ranking)
    - **Status**: Investment signal based on historical performance

    **Interpretation** ({selected_metric_display}):
    """)

    if metric_type in ['ROE', 'ROA']:
        st.markdown("""
        - **Excellent** (Z > 1.5): Significantly above historical average
        - **Good** (0.5 < Z ≤ 1.5): Above historical average
        - **Average** (-0.5 ≤ Z ≤ 0.5): Around historical average
        - **Below Average** (-1.5 ≤ Z < -0.5): Below historical average
        - **Poor** (Z < -1.5): Significantly below historical average
        """)
    else:
        st.markdown("""
        - **Very High** (Z > 1.5): Significantly above historical average
        - **High** (0.5 < Z ≤ 1.5): Above historical average
        - **Average** (-0.5 ≤ Z ≤ 0.5): Around historical average
        - **Low** (-1.5 ≤ Z < -0.5): Below historical average
        - **Very Low** (Z < -1.5): Significantly below historical average
        """)

    st.markdown(f"""
    **Current Dataset**:
    - **Total Records**: {len(df):,}
    - **Date Range**: {df['TRADE_DATE'].min().strftime('%Y-%m-%d')} to {df['TRADE_DATE'].max().strftime('%Y-%m-%d')}
    - **Brokers Analyzed**: {df['TICKER'].nunique()} (including aggregates)
    - **Include Unlisted**: {'Yes' if include_unlisted else 'No'}
    """)