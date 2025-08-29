import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Financial Charts", layout="wide")

# Load IS quarterly data
@st.cache_data
def load_is_data():
    try:
        return pd.read_csv("sql/IS_security_quarterly.csv")
    except Exception as e:
        st.error(f"Error loading IS_security_quarterly.csv: {e}")
        return pd.DataFrame()

df_is = load_is_data()

def get_metric_display_name(metric_code):
    """Convert metric code to display name"""
    metric_names = {
        'ISA1': 'Total Revenue',
        'ISS115': 'Net Interest Income',
        'ISS116': 'Interest Income',
        'ISS117': 'Net Interest Income (Adjusted)',
        'ISS118': 'Fee Income',
        'ISS119': 'Securities Trading Income',
        'ISS120': 'Other Operating Income',
        'ISA2': 'Total Revenue (Adjusted)',
        'ISA4': 'Operating Profit',
        'ISA10': 'Profit Before Tax',
        'ISA23': 'Net Profit After Tax',
        'ISA24': 'Net Profit to Shareholders'
    }
    return metric_names.get(metric_code, metric_code)

def create_quarter_label(row):
    """Create quarter label like Q1 2017"""
    return f"Q{int(row['LENGTHREPORT'])} {int(row['YEARREPORT'])}"

def create_annual_data(df, selected_metrics):
    """Aggregate quarterly data to annual"""
    annual_data = []
    
    # Group by ticker and year, then sum up quarters
    for (ticker, year), group in df.groupby(['TICKER', 'YEARREPORT']):
        annual_row = group.iloc[0].copy()  # Start with first row
        
        # Sum all numeric columns
        for col in selected_metrics:
            if col in group.columns:
                annual_row[col] = group[col].sum()
        
        annual_row['LENGTHREPORT'] = 'Annual'
        annual_row['Quarter_Label'] = f"{int(year)}"
        annual_data.append(annual_row)
    
    return pd.DataFrame(annual_data)

st.title("📊 Financial Charts")
st.markdown("Interactive charts for quarterly and annual financial metrics")

if df_is.empty:
    st.warning("No data available. Please ensure IS_security_quarterly.csv exists in the sql/ directory.")
    st.stop()

# Create quarter labels for plotting
df_is['Quarter_Label'] = df_is.apply(create_quarter_label, axis=1)

# Sidebar filters
st.sidebar.header("🎛️ Filters")

# Broker selection
available_brokers = sorted(df_is['TICKER'].dropna().astype(str).unique())
selected_brokers = st.sidebar.multiselect(
    "Select Brokers:",
    options=available_brokers,
    default=available_brokers[:3] if len(available_brokers) >= 3 else available_brokers
)

# Metric selection
metric_columns = [col for col in df_is.columns if col.startswith(('ISA', 'ISS'))]
selected_metrics = st.sidebar.multiselect(
    "Select Metrics:",
    options=metric_columns,
    default=['ISA1', 'ISA10', 'ISA23'] if 'ISA1' in metric_columns else metric_columns[:3],
    format_func=get_metric_display_name
)

# Year selection
available_years = sorted(df_is['YEARREPORT'].unique())
selected_years = st.sidebar.multiselect(
    "Select Years:",
    options=available_years,
    default=available_years[-3:] if len(available_years) >= 3 else available_years
)

# Quarter selection
available_quarters = sorted(df_is['LENGTHREPORT'].unique())
selected_quarters = st.sidebar.multiselect(
    "Select Quarters:",
    options=available_quarters,
    default=available_quarters,
    format_func=lambda x: f"Q{int(x)}"
)

# Data aggregation option
data_view = st.sidebar.radio(
    "Data View:",
    options=["Quarterly", "Annual", "Both"],
    index=0
)

# Chart type
chart_type = st.sidebar.selectbox(
    "Chart Type:",
    options=["Line Chart", "Bar Chart", "Area Chart"],
    index=0
)

# Filter data
filtered_df = df_is[
    (df_is['TICKER'].isin(selected_brokers)) &
    (df_is['YEARREPORT'].isin(selected_years)) &
    (df_is['LENGTHREPORT'].isin(selected_quarters))
].copy()

if selected_metrics and not filtered_df.empty:
    
    # Prepare data based on view selection
    plot_data = []
    
    if data_view in ["Quarterly", "Both"]:
        quarterly_df = filtered_df.copy()
        quarterly_df['Data_Type'] = 'Quarterly'
        plot_data.append(quarterly_df)
    
    if data_view in ["Annual", "Both"]:
        annual_df = create_annual_data(filtered_df, selected_metrics)
        annual_df['Data_Type'] = 'Annual'
        plot_data.append(annual_df)
    
    if plot_data:
        combined_df = pd.concat(plot_data, ignore_index=True)
        
        # Main content area with tabs for each metric
        tab_names = [get_metric_display_name(metric) for metric in selected_metrics]
        tabs = st.tabs(tab_names)
        
        for i, metric in enumerate(selected_metrics):
            with tabs[i]:
                st.subheader(f"📈 {get_metric_display_name(metric)}")
                
                # Create subplot for each broker
                fig = make_subplots(
                    rows=1, cols=len(selected_brokers),
                    subplot_titles=selected_brokers,
                    shared_yaxes=True
                )
                
                colors = px.colors.qualitative.Set3
                
                for j, broker in enumerate(selected_brokers):
                    broker_data = combined_df[combined_df['TICKER'] == broker].copy()
                    
                    if not broker_data.empty and metric in broker_data.columns:
                        # Convert values to millions for better readability
                        broker_data[f'{metric}_millions'] = pd.to_numeric(broker_data[metric], errors='coerce') / 1_000_000
                        
                        if data_view == "Both":
                            # Separate quarterly and annual data
                            quarterly_data = broker_data[broker_data['Data_Type'] == 'Quarterly']
                            annual_data = broker_data[broker_data['Data_Type'] == 'Annual']
                            
                            if not quarterly_data.empty:
                                # Sort quarterly data chronologically
                                quarterly_data = quarterly_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])
                                
                                if chart_type == "Line Chart":
                                    fig.add_trace(
                                        go.Scatter(
                                            x=quarterly_data['Quarter_Label'],
                                            y=quarterly_data[f'{metric}_millions'],
                                            name=f"{broker} (Quarterly)",
                                            line=dict(color=colors[j % len(colors)], width=2),
                                            mode='lines+markers'
                                        ),
                                        row=1, col=j+1
                                    )
                                elif chart_type == "Bar Chart":
                                    fig.add_trace(
                                        go.Bar(
                                            x=quarterly_data['Quarter_Label'],
                                            y=quarterly_data[f'{metric}_millions'],
                                            name=f"{broker} (Quarterly)",
                                            marker_color=colors[j % len(colors)],
                                            opacity=0.7
                                        ),
                                        row=1, col=j+1
                                    )
                                elif chart_type == "Area Chart":
                                    fig.add_trace(
                                        go.Scatter(
                                            x=quarterly_data['Quarter_Label'],
                                            y=quarterly_data[f'{metric}_millions'],
                                            name=f"{broker} (Quarterly)",
                                            fill='tozeroy',
                                            fillcolor=colors[j % len(colors)],
                                            line=dict(color=colors[j % len(colors)])
                                        ),
                                        row=1, col=j+1
                                    )
                            
                            if not annual_data.empty:
                                # Sort annual data
                                annual_data = annual_data.sort_values('YEARREPORT')
                                
                                fig.add_trace(
                                    go.Scatter(
                                        x=annual_data['Quarter_Label'],
                                        y=annual_data[f'{metric}_millions'],
                                        name=f"{broker} (Annual)",
                                        line=dict(color=colors[j % len(colors)], width=3, dash='dash'),
                                        mode='lines+markers',
                                        marker=dict(size=8)
                                    ),
                                    row=1, col=j+1
                                )
                        else:
                            # Single data type
                            broker_data = broker_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])
                            
                            x_values = broker_data['Quarter_Label'] if data_view == "Quarterly" else broker_data['Quarter_Label']
                            
                            if chart_type == "Line Chart":
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_values,
                                        y=broker_data[f'{metric}_millions'],
                                        name=broker,
                                        line=dict(color=colors[j % len(colors)], width=2),
                                        mode='lines+markers'
                                    ),
                                    row=1, col=j+1
                                )
                            elif chart_type == "Bar Chart":
                                fig.add_trace(
                                    go.Bar(
                                        x=x_values,
                                        y=broker_data[f'{metric}_millions'],
                                        name=broker,
                                        marker_color=colors[j % len(colors)]
                                    ),
                                    row=1, col=j+1
                                )
                            elif chart_type == "Area Chart":
                                fig.add_trace(
                                    go.Scatter(
                                        x=x_values,
                                        y=broker_data[f'{metric}_millions'],
                                        name=broker,
                                        fill='tozeroy',
                                        fillcolor=colors[j % len(colors)],
                                        line=dict(color=colors[j % len(colors)])
                                    ),
                                    row=1, col=j+1
                                )
                
                # Update layout
                fig.update_layout(
                    height=500,
                    title=f"{get_metric_display_name(metric)} - {data_view} View",
                    showlegend=True,
                    hovermode='x unified'
                )
                
                fig.update_yaxes(title_text="Value (Millions VND)")
                fig.update_xaxes(title_text="Period")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Summary Statistics")
                    summary_stats = []
                    for broker in selected_brokers:
                        broker_data = combined_df[
                            (combined_df['TICKER'] == broker) & 
                            (pd.to_numeric(combined_df[metric], errors='coerce').notna())
                        ]
                        if not broker_data.empty:
                            values = pd.to_numeric(broker_data[metric], errors='coerce') / 1_000_000
                            summary_stats.append({
                                'Broker': broker,
                                'Mean': f"{values.mean():.1f}M",
                                'Max': f"{values.max():.1f}M",
                                'Min': f"{values.min():.1f}M",
                                'Latest': f"{values.iloc[-1]:.1f}M"
                            })
                    
                    if summary_stats:
                        st.dataframe(pd.DataFrame(summary_stats), hide_index=True)
                
                with col2:
                    st.subheader("📋 Data Preview")
                    preview_data = combined_df[
                        combined_df['TICKER'].isin(selected_brokers)
                    ][['TICKER', 'Quarter_Label', metric, 'Data_Type']].copy()
                    
                    if not preview_data.empty:
                        preview_data[f'{metric}_millions'] = pd.to_numeric(preview_data[metric], errors='coerce') / 1_000_000
                        preview_data = preview_data.drop(columns=[metric])
                        preview_data = preview_data.rename(columns={
                            'TICKER': 'Broker',
                            'Quarter_Label': 'Period',
                            f'{metric}_millions': f'{get_metric_display_name(metric)} (M)',
                            'Data_Type': 'Type'
                        })
                        
                        st.dataframe(
                            preview_data.sort_values(['Broker', 'Period']).head(20),
                            hide_index=True
                        )

else:
    st.warning("Please select brokers and metrics to display charts.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar filters to customize your charts. Values are displayed in millions of VND for better readability.")