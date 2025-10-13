import math
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import toml

from utils.keycode_matcher import load_keycode_map, match_keycodes


st.set_page_config(page_title="Forecast", layout="wide")

if st.sidebar.button("Reload Data"):
    st.cache_data.clear()


@st.cache_data
def load_data():
    """Load quarterly actuals, index data, and full-year forecast data."""

    theme_config = toml.load("utils/config.toml")

    keycode_map = load_keycode_map('sql/IRIS_KEYCODE.csv')
    df_is_quarterly = match_keycodes('sql/IS_security_quarterly.csv', keycode_map)

    df_forecast = pd.read_csv('sql/FORECAST.csv', low_memory=False)
    df_forecast['DATE'] = pd.to_numeric(df_forecast['DATE'], errors='coerce')
    df_forecast = df_forecast.dropna(subset=['DATE'])
    df_forecast['DATE'] = df_forecast['DATE'].astype(int)
    df_forecast['VALUE'] = pd.to_numeric(df_forecast['VALUE'], errors='coerce')

    df_index = pd.read_csv('sql/INDEX.csv', parse_dates=['TRADINGDATE'])
    df_turnover = pd.read_excel('sql/turnover.xlsx')

    return theme_config, df_is_quarterly, df_forecast, df_index, df_turnover


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


def prev_quarter(year: int, quarter: int) -> tuple[int, int]:
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


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


theme_config, df_is_quarterly, df_forecast, df_index_raw, df_turnover = load_data()

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

# Prepare market turnover metrics by quarter
df_index = df_index_raw.copy()
if 'COMGROUPCODE' in df_index.columns:
    df_index = df_index[df_index['COMGROUPCODE'] == 'VNINDEX']
df_index['TRADINGDATE'] = pd.to_datetime(df_index['TRADINGDATE'], errors='coerce')
df_index = df_index.dropna(subset=['TRADINGDATE'])

if not df_index.empty and 'TOTALVALUE' in df_index.columns:
    df_index['QuarterPeriod'] = df_index['TRADINGDATE'].dt.to_period('Q')
    quarter_stats_df = df_index.groupby('QuarterPeriod').agg(
        total_turnover=('TOTALVALUE', 'sum'),
        trading_days=('TRADINGDATE', 'nunique')
    ).reset_index()
    quarter_stats_df['Year'] = quarter_stats_df['QuarterPeriod'].dt.year
    quarter_stats_df['Quarter'] = quarter_stats_df['QuarterPeriod'].dt.quarter
    quarter_stats_df['avg_daily_bn'] = quarter_stats_df.apply(
        lambda row: (row['total_turnover'] / row['trading_days'] / 1e9)
        if row['trading_days'] else np.nan,
        axis=1
    )
else:
    quarter_stats_df = pd.DataFrame(columns=['Year', 'Quarter', 'total_turnover', 'trading_days', 'avg_daily_bn'])

quarter_stats_lookup = {
    (int(row['Year']), int(row['Quarter'])): {
        'total_turnover': float(row['total_turnover']) if not pd.isna(row['total_turnover']) else None,
        'trading_days': int(row['trading_days']) if not pd.isna(row['trading_days']) else None,
        'avg_daily_bn': float(row['avg_daily_bn']) if not pd.isna(row['avg_daily_bn']) else None,
    }
    for _, row in quarter_stats_df.iterrows()
}

turnover_share_lookup = {}
if not df_turnover.empty:
    turnover_filtered = df_turnover[df_turnover['Ticker'] == selected_broker].copy()
    turnover_filtered['share'] = turnover_filtered.apply(
        lambda row: (row['Company turnover'] / row['Market turnover'])
        if row['Market turnover'] not in (0, None, np.nan) else np.nan,
        axis=1
    )
    turnover_share_lookup = {
        int(row['Year']): float(row['share'])
        for _, row in turnover_filtered.iterrows()
        if not pd.isna(row['share'])
    }

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

if missing_segments:
    st.info("Missing forecast values for: " + ", ".join(missing_segments))

# Brokerage segment detailed table (last 3 quarters + forecast)
brokerage_history_quarters = []
year_cursor, quarter_cursor = target_year, target_quarter
for _ in range(3):
    year_cursor, quarter_cursor = prev_quarter(year_cursor, quarter_cursor)
    brokerage_history_quarters.append((year_cursor, quarter_cursor))
brokerage_history_quarters = list(reversed(brokerage_history_quarters))

def get_share_for_year(year: int) -> float | None:
    if year in turnover_share_lookup:
        return turnover_share_lookup[year]
    if not turnover_share_lookup:
        return None
    years_sorted = sorted(turnover_share_lookup.keys())
    prior_years = [y for y in years_sorted if y <= year]
    if prior_years:
        return turnover_share_lookup[prior_years[-1]]
    future_years = [y for y in years_sorted if y >= year]
    if future_years:
        return turnover_share_lookup[future_years[0]]
    return None

@st.cache_data(ttl=3600)
def fetch_market_share_api(broker: str, year: int, quarter: int):
    try:
        url = "https://api.hsx.vn/s/api/v1/1/brokeragemarketshare/top/ten"
        params = {
            'pageIndex': 1,
            'pageSize': 30,
            'year': year,
            'period': quarter,
            'dateType': 1
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('success') and 'data' in data:
            for item in data['data'].get('brokerageStock', []):
                if item.get('shortenName', '').upper() == broker.upper():
                    return float(item.get('percentage', 0))
    except Exception:
        return None
    return None

history_metrics = []
year_fee_observations = {}

for y, q in brokerage_history_quarters:
    stats = quarter_stats_lookup.get((y, q), {})
    total_turnover = stats.get('total_turnover')
    trading_days = stats.get('trading_days')
    avg_daily_bn = stats.get('avg_daily_bn')

    quarter_row = df_quarters[
        (df_quarters['YEARREPORT'] == y) & (df_quarters['LENGTHREPORT'] == q)
    ]
    net_brokerage = float(quarter_row['brokerage_fee'].iloc[0]) if not quarter_row.empty else None

    api_share_pct = fetch_market_share_api(selected_broker, y, q)

    share_decimal = None
    share_pct_display = None
    if api_share_pct:
        share_decimal = (api_share_pct / 100) / 2
        share_pct_display = share_decimal * 100
    else:
        share_year = get_share_for_year(y)
        if share_year:
            share_decimal = share_year / 2
            share_pct_display = share_decimal * 100

    fee_decimal = None
    if (
        share_decimal
        and share_decimal > 0
        and total_turnover not in (None, 0)
        and net_brokerage not in (None, 0)
    ):
        fee_decimal = net_brokerage / (total_turnover * 2 * share_decimal)
        if fee_decimal <= 0:
            fee_decimal = None

    if fee_decimal:
        year_fee_observations.setdefault(y, []).append(fee_decimal)

    history_metrics.append({
        'label': quarter_label(y, q),
        'year': y,
        'quarter': q,
        'avg_daily_bn': avg_daily_bn,
        'share_pct': share_pct_display,
        'fee_bps': fee_decimal * 10000 if fee_decimal else None,
        'net_brokerage_bn': net_brokerage / 1e9 if net_brokerage is not None else None,
        'total_turnover': total_turnover,
        'trading_days': trading_days,
        'net_brokerage': net_brokerage,
        'share_decimal': share_decimal,
    })

history_avg_daily = [m['avg_daily_bn'] for m in history_metrics if m['avg_daily_bn'] is not None]
history_fee_bps = [m['fee_bps'] for m in history_metrics if m['fee_bps'] is not None]
history_trading_days = [m['trading_days'] for m in history_metrics if m['trading_days']]

fee_decimal_default = None
if year_fee_observations:
    latest_year = max(year_fee_observations.keys())
    fees = [f for f in year_fee_observations[latest_year] if f]
    if fees:
        fee_decimal_default = float(np.mean(fees))

share_default = None
last_with_share = next((m for m in reversed(history_metrics) if m['share_decimal'] is not None), None)
if last_with_share:
    share_default = last_with_share['share_decimal']
if share_default is None:
    share_year = get_share_for_year(target_year)
    if share_year is not None:
        share_default = share_year / 2
if share_default is None:
    share_default = 0.05

avg_daily_default = history_avg_daily[-1] if history_avg_daily else 0.0
if fee_decimal_default:
    fee_default = fee_decimal_default * 10000
elif history_fee_bps:
    fee_default = history_fee_bps[-1]
else:
    fee_default = 2.0

trading_days_forecast = int(round(np.mean(history_trading_days))) if history_trading_days else 63

share_default_pct = share_default * 100

brokerage_input_cols = st.columns(3)
avg_daily_initial = float(round(avg_daily_default)) if avg_daily_default is not None else 0.0
avg_daily_input = brokerage_input_cols[0].number_input(
    f"{target_label} VNI Avg Daily Turnover (bn VND)",
    value=avg_daily_initial,
    min_value=0.0,
    step=50.0,
    format="%.0f"
)
market_share_initial = float(round(share_default_pct, 2)) if share_default_pct is not None else 0.0
market_share_input = brokerage_input_cols[1].number_input(
    f"{target_label} Market Share (%)",
    value=market_share_initial,
    min_value=0.0,
    step=0.1,
    format="%.2f"
)
net_fee_initial = float(round(fee_default, 2)) if fee_default is not None else 0.0
net_fee_input = brokerage_input_cols[2].number_input(
    f"{target_label} Net Brokerage Fee (bps)",
    value=net_fee_initial,
    min_value=0.0,
    step=0.1,
    format="%.2f"
)

market_share_decimal = market_share_input / 100
net_fee_decimal = net_fee_input / 10000
total_turnover_forecast = avg_daily_input * 1e9 * trading_days_forecast
net_brokerage_forecast = total_turnover_forecast * 2 * market_share_decimal * net_fee_decimal
net_brokerage_forecast_bn = net_brokerage_forecast / 1e9

forecast_metrics = {
    'label': target_label,
    'avg_daily_bn': avg_daily_input,
    'share_pct': market_share_input,
    'fee_bps': net_fee_input,
    'net_brokerage_bn': net_brokerage_forecast_bn,
}

def fmt_value(value, decimals=0, suffix=""):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.{decimals}f}{suffix}"

table_rows = {
    'Metric': [
        'VNI Avg Trading Value (bn/day)',
        'Market Share (%)',
        'Net Brokerage Fee (bps)',
        'Net Brokerage Income (bn)',
    ]
}

for metrics in history_metrics:
    table_rows[metrics['label']] = [
        fmt_value(metrics['avg_daily_bn']),
        fmt_value(metrics['share_pct'], 2),
        fmt_value(metrics['fee_bps'], 2),
        fmt_value(metrics['net_brokerage_bn']),
    ]

table_rows[forecast_metrics['label']] = [
    fmt_value(forecast_metrics['avg_daily_bn']),
    fmt_value(forecast_metrics['share_pct'], 2),
    fmt_value(forecast_metrics['fee_bps'], 2),
    fmt_value(forecast_metrics['net_brokerage_bn']),
]

brokerage_table_df = pd.DataFrame(table_rows)
brokerage_table_df = brokerage_table_df.set_index('Metric')
st.dataframe(brokerage_table_df, use_container_width=True)
st.caption(f"Assuming {trading_days_forecast} trading days for {target_label} and applying net brokerage formula.")

# Update summary with forecast brokerage
target_column_label = f"{target_label} Base (bn VND)"
if target_column_label in summary_df.columns:
    summary_df.loc[summary_df['Segment'] == 'Brokerage Fee', target_column_label] = format_bn_str(net_brokerage_forecast)

st.subheader("Baseline Breakdown")
st.caption(f"Base assumptions derived from {target_year} full-year forecast minus actual results up to {latest_label}.")
st.dataframe(summary_df, hide_index=True)

st.subheader(f"{target_label} Segment Assumptions")

segment_inputs = {}
input_columns = st.columns(3)

for idx, segment in enumerate(SEGMENTS):
    if segment['key'] == 'brokerage_fee':
        continue
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

segment_inputs['brokerage_fee'] = net_brokerage_forecast

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
    top: 60px;
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
    height: 100px;
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
