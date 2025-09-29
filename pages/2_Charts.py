import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Financial Charts", layout="wide")

# Load financial statement data
@st.cache_data
def load_financial_data():
    """Load IS, BS, and Note security data"""
    data = {}
    
    # Load Income Statement data
    try:
        data['IS'] = pd.read_csv("sql/IS_security.csv")
    except Exception as e:
        st.error(f"Error loading IS_security.csv: {e}")
        data['IS'] = pd.DataFrame()
    
    # Load Balance Sheet data
    try:
        data['BS'] = pd.read_csv("sql/BS_security.csv")
    except Exception as e:
        st.error(f"Error loading BS_security.csv: {e}")
        data['BS'] = pd.DataFrame()
    
    # Load Note data
    try:
        data['NOTE'] = pd.read_csv("sql/Note_security.csv")
    except Exception as e:
        st.error(f"Error loading Note_security.csv: {e}")
        data['NOTE'] = pd.DataFrame()
    
    return data

financial_data = load_financial_data()

def get_metric_display_name(metric_code):
    """Convert metric code to display name"""
    metric_names = {
        # Income Statement metrics
        'ISA1': 'Total Revenue',
        'ISA4': 'Operating Profit',
        'ISA10': 'Profit Before Tax',
        'ISA23': 'Net Profit After Tax',
        'ISA24': 'Net Profit to Shareholders',
        'ISS115': 'Net Interest Income',
        'ISS116': 'Interest Income',
        'ISS118': 'Fee Income',
        'ISS119': 'Securities Trading Income',
        
        # Balance Sheet metrics
        'BSA1': 'Total Assets',
        'BSA2': 'Current Assets',
        'BSA3': 'Cash and Cash Equivalents',
        'BSA4': 'Short-term Investments',
        'BSA5': 'Short-term Receivables',
        'BSA10': 'Total Liabilities',
        'BSA11': 'Short-term Liabilities',
        'BSA12': 'Long-term Liabilities'
    }
    return metric_names.get(metric_code, metric_code)

def create_quarter_label(row):
    """Create quarter label like 2025Q1"""
    quarter_map = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4', 6: 'H1', 9: '9M'}
    quarter = quarter_map.get(int(row['LENGTHREPORT']), f"Q{int(row['LENGTHREPORT'])}")
    return f"{int(row['YEARREPORT'])}{quarter}"

def create_annual_data(df, selected_metrics):
    """Aggregate quarterly data to annual"""
    annual_data = []
    
    # Group by ticker and year, then sum up quarters
    for (ticker, year), group in df.groupby(['TICKER', 'YEARREPORT']):
        # Only include if we have Q4 data or annual data
        if 4 in group['LENGTHREPORT'].values or len(group) >= 4:
            annual_row = group.iloc[0].copy()  # Start with first row
            
            # Sum all numeric columns
            for col in selected_metrics:
                if col in group.columns:
                    annual_row[col] = pd.to_numeric(group[col], errors='coerce').sum()
            
            annual_row['LENGTHREPORT'] = 'Annual'
            annual_row['Quarter_Label'] = f"{int(year)}"
            annual_data.append(annual_row)
    
    return pd.DataFrame(annual_data)

def get_available_metrics(data_dict):
    """Get available metrics from all datasets"""
    metrics = {'IS': [], 'BS': [], 'NOTE': []}
    
    for key, df in data_dict.items():
        if not df.empty:
            if key == 'IS':
                metrics['IS'] = [col for col in df.columns if col.startswith('IS') and col not in ['ISAUDIT']]
            elif key == 'BS':
                metrics['BS'] = [col for col in df.columns if col.startswith('BS')]
            elif key == 'NOTE':
                metrics['NOTE'] = [col for col in df.columns if col.startswith('NOTE') and col not in ['NOTESECURITYID', 'NOTE']]
    
    return metrics

st.title("📊 Financial Charts")
st.markdown("Interactive charts for quarterly and annual financial metrics")

# Check if data is available
if all(df.empty for df in financial_data.values()):
    st.warning("No data available. Please ensure IS_security.csv, BS_security.csv, and Note_security.csv exist in the sql/ directory.")
    st.stop()

# Get available metrics
available_metrics = get_available_metrics(financial_data)

# Combine all data for unified processing
combined_data = []
for data_type, df in financial_data.items():
    if not df.empty:
        df_copy = df.copy()
        df_copy['Data_Source'] = data_type
        df_copy['Quarter_Label'] = df_copy.apply(create_quarter_label, axis=1)
        combined_data.append(df_copy)

if combined_data:
    df_combined = pd.concat(combined_data, ignore_index=True, sort=False)
else:
    df_combined = pd.DataFrame()

# Sidebar filters
st.sidebar.header("🎛️ Filters")

if df_combined.empty:
    st.warning("No combined data available for filtering.")
    st.stop()

# Broker selection
available_brokers = sorted([str(b) for b in df_combined['TICKER'].unique() if pd.notna(b)])
selected_brokers = st.sidebar.multiselect(
    "Select Brokers:",
    options=available_brokers,
    default=available_brokers[:3] if len(available_brokers) >= 3 else available_brokers
)

# Data source selection
data_source = st.sidebar.selectbox(
    "Select Data Source:",
    options=['IS', 'BS', 'NOTE'],
    format_func=lambda x: {'IS': 'Income Statement', 'BS': 'Balance Sheet', 'NOTE': 'Notes'}.get(x, x)
)

# Metric selection based on data source
if data_source in available_metrics and available_metrics[data_source]:
    selected_metrics = st.sidebar.multiselect(
        "Select Metrics:",
        options=available_metrics[data_source],
        default=available_metrics[data_source][:3] if len(available_metrics[data_source]) >= 3 else available_metrics[data_source],
        format_func=get_metric_display_name
    )
else:
    selected_metrics = []
    st.sidebar.warning(f"No metrics available for {data_source}")

# Year selection
available_years = sorted(df_combined['YEARREPORT'].unique())
selected_years = st.sidebar.multiselect(
    "Select Years:",
    options=available_years,
    default=available_years[-3:] if len(available_years) >= 3 else available_years
)

# Data aggregation option
data_view = st.sidebar.radio(
    "Data View:",
    options=["Quarterly", "Annual"],
    index=0
)

# Filter data
filtered_df = df_combined[
    (df_combined['TICKER'].isin(selected_brokers)) &
    (df_combined['YEARREPORT'].isin(selected_years)) &
    (df_combined['Data_Source'] == data_source)
].copy()

# Further filter by quarters for quarterly view
if data_view == "Quarterly":
    # Only include quarterly data (1-4)
    filtered_df = filtered_df[filtered_df['LENGTHREPORT'].isin([1, 2, 3, 4])]
else:
    # For annual view, aggregate the data
    if not filtered_df.empty and selected_metrics:
        filtered_df = create_annual_data(filtered_df, selected_metrics)
        if not filtered_df.empty:
            filtered_df['Data_Source'] = data_source

if selected_metrics and not filtered_df.empty:
    # Main content area with tabs for each metric
    tab_names = [get_metric_display_name(metric) for metric in selected_metrics]
    tabs = st.tabs(tab_names)
    
    for i, metric in enumerate(selected_metrics):
        with tabs[i]:
            st.subheader(f"📈 {get_metric_display_name(metric)}")
            
            # Create line chart
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            
            for j, broker in enumerate(selected_brokers):
                broker_data = filtered_df[filtered_df['TICKER'] == broker].copy()
                
                if not broker_data.empty and metric in broker_data.columns:
                    # Convert values to millions for better readability
                    broker_data = broker_data.sort_values(['YEARREPORT', 'LENGTHREPORT'])
                    
                    # Clean and convert numeric values
                    values = pd.to_numeric(broker_data[metric], errors='coerce') / 1_000_000
                    broker_data[f'{metric}_millions'] = values
                    
                    # Remove NaN values for cleaner charts
                    clean_data = broker_data.dropna(subset=[f'{metric}_millions'])
                    
                    if not clean_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=clean_data['Quarter_Label'],
                                y=clean_data[f'{metric}_millions'],
                                name=broker,
                                line=dict(color=colors[j % len(colors)], width=2),
                                mode='lines+markers',
                                marker=dict(size=6)
                            )
                        )
            
            # Update layout
            fig.update_layout(
                height=500,
                title=f"{get_metric_display_name(metric)} - {data_view} View",
                xaxis_title="Period",
                yaxis_title="Value (Millions VND)",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data preview
            st.subheader("📋 Data Preview")
            preview_data = filtered_df[
                (filtered_df['TICKER'].isin(selected_brokers)) &
                (pd.to_numeric(filtered_df[metric], errors='coerce').notna())
            ][['TICKER', 'Quarter_Label', metric]].copy()
            
            if not preview_data.empty:
                preview_data[f'{metric}_millions'] = pd.to_numeric(preview_data[metric], errors='coerce') / 1_000_000
                preview_data = preview_data.drop(columns=[metric])
                preview_data = preview_data.rename(columns={
                    'TICKER': 'Broker',
                    'Quarter_Label': 'Period',
                    f'{metric}_millions': f'{get_metric_display_name(metric)} (M)'
                })
                
                st.dataframe(
                    preview_data.sort_values(['Broker', 'Period']),
                    hide_index=True
                )

else:
    st.warning("Please select brokers and metrics to display charts.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar filters to customize your charts. Values are displayed in millions of VND for better readability.")