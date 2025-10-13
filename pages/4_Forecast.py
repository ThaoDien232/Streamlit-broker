import math
from datetime import datetime

import pandas as pd
import streamlit as st
import toml

from utils.keycode_matcher import load_keycode_map, match_keycodes


st.set_page_config(page_title="Forecast", layout="wide")

if st.sidebar.button("Reload Data"):
    st.cache_data.clear()


@st.cache_data
def load_data():
    """Load quarterly actuals and full-year forecast data."""

    theme_config = toml.load("utils/config.toml")

    keycode_map = load_keycode_map('sql/IRIS_KEYCODE.csv')
    df_is_quarterly = match_keycodes('sql/IS_security_quarterly.csv', keycode_map)

    df_forecast = pd.read_csv('sql/FORECAST.csv', low_memory=False)
    df_forecast['DATE'] = pd.to_numeric(df_forecast['DATE'], errors='coerce')
    df_forecast = df_forecast.dropna(subset=['DATE'])
    df_forecast['DATE'] = df_forecast['DATE'].astype(int)
    df_forecast['VALUE'] = pd.to_numeric(df_forecast['VALUE'], errors='coerce')

    return theme_config, df_is_quarterly, df_forecast


SEGMENTS = [
    {
        "key": "brokerage_fee",
        "label": "Brokerage Fee",
        "forecast_key": "Net_Brokerage_Income",
        "columns": ['IS.10', 'IS.33'],
    },
    {
        "key": "margin_income",
        "label": "Margin Income",
        "forecast_key": "Net_Margin_lending_Income",
        "columns": ['IS.7', 'IS.30'],
    },
    {
        "key": "investment_income",
        "label": "Investment Income",
        "forecast_key": "Net_Investment",
        "columns": [
            'IS.3', 'IS.4', 'IS.5', 'IS.8', 'IS.9', 'IS.24', 'IS.25', 'IS.26',
            'IS.27', 'IS.28', 'IS.29', 'IS.31', 'IS.32', 'IS.6'
        ],
    },
    {
        "key": "ib_income",
        "label": "IB Income",
        "forecast_key": "Net_IB_Income",
        "columns": ['IS.11', 'IS.12', 'IS.13', 'IS.15', 'IS.16', 'IS.17', 'IS.18', 'IS.34', 'IS.35', 'IS.36', 'IS.38'],
    },
    {
        "key": "sga",
        "label": "SG&A",
        "forecast_key": "SG_A",
        "columns": ['IS.57', 'IS.58'],
    },
    {
        "key": "interest_expense",
        "label": "Interest Expense",
        "forecast_key": "Interest_Expense",
        "columns": ['IS.51'],
    },
]


def sum_columns(df: pd.DataFrame, columns):
    values = pd.Series(0.0, index=df.index)
    for col in columns:
        if col in df.columns:
            values = values.add(pd.to_numeric(df[col], errors='coerce').fillna(0.0), fill_value=0.0)
    return values


def prepare_quarter_metrics(df_is: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df_is[df_is['TICKER'] == ticker].copy()
    if df.empty:
        return pd.DataFrame()

    df['YEARREPORT'] = pd.to_numeric(df['YEARREPORT'], errors='coerce')
    df['LENGTHREPORT'] = pd.to_numeric(df['LENGTHREPORT'], errors='coerce')
    df = df.dropna(subset=['YEARREPORT', 'LENGTHREPORT'])
    df = df[df['LENGTHREPORT'].isin([1, 2, 3, 4])]
    if df.empty:
        return pd.DataFrame()

    df['YEARREPORT'] = df['YEARREPORT'].astype(int)
    df['LENGTHREPORT'] = df['LENGTHREPORT'].astype(int)
    df['ENDDATE'] = pd.to_datetime(df['ENDDATE'], errors='coerce')

    for segment in SEGMENTS:
        df[segment['key']] = sum_columns(df, segment['columns'])

    df['pbt'] = pd.to_numeric(df.get('IS.65', 0.0), errors='coerce').fillna(0.0)

    columns = ['YEARREPORT', 'LENGTHREPORT', 'ENDDATE', 'pbt'] + [segment['key'] for segment in SEGMENTS]
    return df[columns]


def next_quarter(year: int, quarter: int) -> tuple[int, int]:
    if quarter == 4:
        return year + 1, 1
    return year, quarter + 1


def quarter_label(year: int, quarter: int) -> str:
    return f"{year} Q{quarter}"


def collect_totals(df: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    subset = df.loc[mask]
    totals = {segment['key']: subset[segment['key']].sum() if not subset.empty else 0.0 for segment in SEGMENTS}
    totals['pbt'] = subset['pbt'].sum() if not subset.empty else 0.0
    return totals


def get_forecast_map(df_forecast: pd.DataFrame, ticker: str, year: int, keys: list[str]) -> dict[str, float]:
    subset = df_forecast[
        (df_forecast['TICKER'] == ticker) &
        (df_forecast['DATE'] == year) &
        (df_forecast['KEYCODE'].isin(keys))
    ]

    if subset.empty:
        return {}

    grouped = subset.groupby('KEYCODE')['VALUE'].sum().dropna()
    return grouped.to_dict()


def format_bn(value: float) -> float:
    return value / 1e9 if value else 0.0


def format_bn_str(value: float) -> str:
    return f"{format_bn(value):,.0f}"


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:,.1f}%"


theme_config, df_is_quarterly, df_forecast = load_data()

theme = theme_config.get("theme", {}) if isinstance(theme_config, dict) else {}
background_color = theme.get("backgroundColor", "#FFFFFF")
text_color = theme.get("textColor", "#000000")
primary_color = theme.get("primaryColor", "#FF4B4B")

available_brokers = sorted(df_is_quarterly['TICKER'].dropna().unique())
if not available_brokers:
    st.error("No brokerage data available.")
    st.stop()

default_broker = 'SSI' if 'SSI' in available_brokers else available_brokers[0]
selected_broker = st.sidebar.selectbox("Select Broker", options=available_brokers, index=available_brokers.index(default_broker))

df_quarters = prepare_quarter_metrics(df_is_quarterly, selected_broker)
if df_quarters.empty:
    st.warning("No quarterly data found for the selected broker.")
    st.stop()

today = pd.Timestamp.today().normalize()
df_with_dates = df_quarters.dropna(subset=['ENDDATE'])

if not df_with_dates.empty:
    df_actual = df_with_dates[df_with_dates['ENDDATE'] <= today]
    if df_actual.empty:
        df_actual = df_with_dates.copy()
else:
    df_actual = df_quarters.copy()

df_actual = df_actual.sort_values(['YEARREPORT', 'LENGTHREPORT'])
latest_row = df_actual.iloc[-1]
latest_year = int(latest_row['YEARREPORT'])
latest_quarter = int(latest_row['LENGTHREPORT'])

target_year, target_quarter = next_quarter(latest_year, latest_quarter)
latest_label = quarter_label(latest_year, latest_quarter)
target_label = quarter_label(target_year, target_quarter)

forecast_keys = [segment['forecast_key'] for segment in SEGMENTS] + ['PBT']
forecast_map = get_forecast_map(df_forecast, selected_broker, target_year, forecast_keys)

ytd_mask = (df_quarters['YEARREPORT'] == target_year) & (df_quarters['LENGTHREPORT'] < target_quarter)
ytd_totals = collect_totals(df_quarters, ytd_mask)

base_segments = {}
missing_segments = []

remaining_quarters = 5 - target_quarter

for segment in SEGMENTS:
    fy_value = forecast_map.get(segment['forecast_key'])
    if fy_value is None or math.isnan(fy_value):
        missing_segments.append(segment['label'])
        fy_value = 0.0

    realized = ytd_totals.get(segment['key'], 0.0)
    base_segments[segment['key']] = (fy_value - realized) / remaining_quarters

fy_pbt = forecast_map.get('PBT', 0.0)
base_pbt = (fy_pbt - ytd_totals.get('pbt', 0.0)) / remaining_quarters

sum_base_segments = sum(base_segments.values())
residual_other = base_pbt - sum_base_segments

prev_pbt = float(latest_row['pbt'])
yoy_row = df_quarters[(df_quarters['YEARREPORT'] == target_year - 1) & (df_quarters['LENGTHREPORT'] == target_quarter)]
yoy_pbt = float(yoy_row['pbt'].iloc[0]) if not yoy_row.empty else None


sticky_header_placeholder = st.empty()

summary_rows = []
quarter_columns = []
for q in range(1, target_quarter):
    quarter_columns.append((q, f"{target_year} Q{q} (bn VND)"))

for segment in SEGMENTS:
    fy_val = forecast_map.get(segment['forecast_key'], 0.0)
    base_val = base_segments[segment['key']]
    record = {
        "Segment": segment['label'],
        "FY Forecast (bn VND)": format_bn_str(fy_val),
        f"{target_label} Base (bn VND)": format_bn_str(base_val),
    }

    for quarter_num, column_label in quarter_columns:
        quarter_row = df_quarters[
            (df_quarters['YEARREPORT'] == target_year) &
            (df_quarters['LENGTHREPORT'] == quarter_num)
        ]
        quarter_value = float(quarter_row[segment['key']].iloc[0]) if not quarter_row.empty else 0.0
        record[column_label] = format_bn_str(quarter_value)

    summary_rows.append(record)

record_pbt = {
    "Segment": "PBT",
    "FY Forecast (bn VND)": format_bn_str(fy_pbt),
    f"{target_label} Base (bn VND)": format_bn_str(base_pbt),
}

for quarter_num, column_label in quarter_columns:
    quarter_row = df_quarters[
        (df_quarters['YEARREPORT'] == target_year) &
        (df_quarters['LENGTHREPORT'] == quarter_num)
    ]
    quarter_value = float(quarter_row['pbt'].iloc[0]) if not quarter_row.empty else 0.0
    record_pbt[column_label] = format_bn_str(quarter_value)

summary_rows.append(record_pbt)

columns_order = ["Segment", "FY Forecast (bn VND)"] + [label for _, label in quarter_columns] + [f"{target_label} Base (bn VND)"]
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df[columns_order]

st.markdown("### Quarterly Forecast Adjustments")
st.caption(f"Base assumptions derived from {target_year} full-year forecast minus actual results up to {latest_label}.")

if missing_segments:
    st.info("Missing forecast values for: " + ", ".join(missing_segments))

st.subheader("Baseline Breakdown")
st.dataframe(summary_df, hide_index=True)

st.subheader(f"{target_label} Segment Assumptions")

segment_inputs = {}
input_columns = st.columns(3)

for idx, segment in enumerate(SEGMENTS):
    col = input_columns[idx % len(input_columns)]
    with col:
        base_bn = format_bn(base_segments.get(segment['key'], 0.0))
        if not math.isfinite(base_bn):
            base_bn = 0.0
        value = col.number_input(
            f"{segment['label']} (bn VND)",
            value=float(round(base_bn)),
            step=10.0,
            format="%.0f"
        )
        segment_inputs[segment['key']] = value * 1e9

adjusted_total_segments = sum(segment_inputs.values())
adjusted_pbt = residual_other + adjusted_total_segments
impact_vs_base = adjusted_pbt - base_pbt
impact_vs_prev = adjusted_pbt - prev_pbt
impact_vs_yoy = None if yoy_pbt is None else adjusted_pbt - yoy_pbt

qoq_pct = None if prev_pbt == 0 else impact_vs_prev / prev_pbt
yoy_pct = None if yoy_pbt in (None, 0) else impact_vs_yoy / yoy_pbt

qoq_value = format_pct(qoq_pct)
qoq_delta = "n/a" if prev_pbt == 0 else f"{impact_vs_prev / 1e9:+,.0f} bn vs {latest_label}"

yoy_value = format_pct(yoy_pct)
yoy_delta = "n/a" if yoy_pbt in (None, 0) else f"{impact_vs_yoy / 1e9:+,.0f} bn vs {quarter_label(target_year - 1, target_quarter)}"

metrics_html = f"""
<style>
#forecast-sticky {{
    position: fixed;
    top: 70px;
    left: 280px;
    right: 0;
    z-index: 1000;
    background-color: {background_color};
    color: {text_color};
    padding: 16px 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}}
.forecast-sticky-inner {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
}}
.forecast-sticky-title {{
    font-size: 26px;
    font-weight: 600;
}}
.forecast-metrics-group {{
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
}}
.forecast-metric {{
    min-width: 180px;
}}
.forecast-metric-label {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {primary_color};
    margin-bottom: 4px;
}}
.forecast-metric-value {{
    font-size: 28px;
    font-weight: 600;
}}
.forecast-metric-delta {{
    font-size: 13px;
    color: rgba(0,0,0,0.6);
}}
.forecast-header-spacer {{
    height: 80px;
}}
@media (max-width: 768px) {{
    #forecast-sticky {{
        left: 0;
        padding: 12px 16px;
    }}
    .forecast-metrics-group {{
        gap: 16px;
    }}
    .forecast-metric-value {{
        font-size: 22px;
    }}
}}
</style>
<div id="forecast-sticky">
    <div class="forecast-sticky-inner">
        <div class="forecast-sticky-title">{selected_broker} – {target_label}</div>
        <div class="forecast-metrics-group">
            <div class="forecast-metric">
                <div class="forecast-metric-label">PBT</div>
                <div class="forecast-metric-value">{adjusted_pbt / 1e9:,.0f} bn</div>
                <div class="forecast-metric-delta">{impact_vs_base / 1e9:+,.0f} vs base</div>
            </div>
            <div class="forecast-metric">
                <div class="forecast-metric-label">QoQ Growth</div>
                <div class="forecast-metric-value">{qoq_value}</div>
                <div class="forecast-metric-delta">{qoq_delta}</div>
            </div>
            <div class="forecast-metric">
                <div class="forecast-metric-label">YoY Growth</div>
                <div class="forecast-metric-value">{yoy_value}</div>
                <div class="forecast-metric-delta">{yoy_delta}</div>
            </div>
        </div>
    </div>
</div>
<div class="forecast-header-spacer"></div>
"""

sticky_header_placeholder.markdown(metrics_html, unsafe_allow_html=True)
