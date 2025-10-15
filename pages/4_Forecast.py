import math
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import toml

from utils.brokerage_codes import get_brokerage_code
from utils.brokerage_data import load_brokerage_metrics
from utils.investment_book import get_investment_data
from utils.market_index_data import load_market_liquidity_data
from utils.db import run_query


st.set_page_config(page_title="Forecast", layout="wide")

if st.sidebar.button("Reload Data"):
    st.cache_data.clear()

if 'price_cache' not in st.session_state:
    st.session_state.price_cache = {}
if 'price_last_updated' not in st.session_state:
    st.session_state.price_last_updated = None


def _safe_read_excel(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_excel(path, **kwargs)
    except FileNotFoundError:
        st.warning(f"Required file '{path}' not found. Proceeding with empty data.")
        return pd.DataFrame()


@st.cache_data
def load_data():
    """Load quarterly actuals, index data, and full-year forecast data."""

    theme_config = toml.load("utils/config.toml")

    df_metrics = load_brokerage_metrics(include_annual=False)

    if df_metrics.empty:
        df_is_quarterly = pd.DataFrame(columns=['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STARTDATE', 'ENDDATE', 'QUARTER_LABEL'])
        df_bs_quarterly = pd.DataFrame(columns=['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STARTDATE', 'ENDDATE', 'QUARTER_LABEL'])
    else:
        df_metrics = df_metrics.copy()
        if 'ACTUAL' in df_metrics.columns:
            df_metrics = df_metrics[df_metrics['ACTUAL'] == 1]

        base_columns = ['TICKER', 'YEARREPORT', 'LENGTHREPORT', 'STARTDATE', 'ENDDATE', 'QUARTER_LABEL']
        df_metrics = df_metrics.drop_duplicates(subset=base_columns + ['KEYCODE'], keep='last')

        required_is_codes = {'PBT', 'NPAT'}
        for segment in SEGMENTS:
            values = segment.get('columns') or []
            if isinstance(values, str):
                required_is_codes.add(values)
            else:
                required_is_codes.update(values)

        df_is = df_metrics[
            df_metrics['KEYCODE'].str.startswith('IS.')
            | df_metrics['KEYCODE'].isin(required_is_codes)
        ]
        if df_is.empty:
            df_is_quarterly = pd.DataFrame(columns=base_columns)
        else:
            df_is_quarterly = (
                df_is.pivot_table(
                    index=base_columns,
                    columns='KEYCODE',
                    values='VALUE',
                    aggfunc='first'
                )
                .reset_index()
            )
            df_is_quarterly.columns.name = None

        df_bs = df_metrics[df_metrics['KEYCODE'].str.startswith('BS.')]
        if df_bs.empty:
            df_bs_quarterly = pd.DataFrame(columns=base_columns)
        else:
            df_bs_quarterly = (
                df_bs.pivot_table(
                    index=base_columns,
                    columns='KEYCODE',
                    values='VALUE',
                    aggfunc='first'
                )
                .reset_index()
            )
            df_bs_quarterly.columns.name = None

    forecast_query = """
        SELECT KEYCODE, KEYCODENAME, ORGANCODE, TICKER, DATE, VALUE, RATING, FORECASTDATE
        FROM SIL.W_F_IRIS_FORECAST
    """

    try:
        df_forecast = run_query(forecast_query)
    except Exception as exc:
        st.warning(f"Database forecast query failed ({exc}); falling back to local CSV if available.")
        df_forecast = pd.DataFrame()

    if df_forecast.empty:
        csv_path = Path('sql/FORECAST.csv')
        if csv_path.exists():
            st.info("Using local 'sql/FORECAST.csv' for forecast data.")
            df_forecast = pd.read_csv(csv_path, low_memory=False)
        else:
            st.warning("Forecast data unavailable from database and local CSV not found. Proceeding without forecasts.")
            df_forecast = pd.DataFrame(columns=['KEYCODE', 'KEYCODENAME', 'ORGANCODE', 'TICKER', 'DATE', 'VALUE', 'RATING', 'FORECASTDATE'])

    if not df_forecast.empty:
        df_forecast['DATE'] = pd.to_numeric(df_forecast['DATE'], errors='coerce')
        df_forecast = df_forecast.dropna(subset=['DATE'])
        df_forecast['DATE'] = df_forecast['DATE'].astype(int)
        df_forecast['VALUE'] = pd.to_numeric(df_forecast['VALUE'], errors='coerce')

    try:
        df_liquidity = load_market_liquidity_data(start_year=2017)
    except Exception as exc:
        st.warning(f"Unable to load market liquidity data from database: {exc}")
        df_liquidity = pd.DataFrame(columns=['Year', 'Quarter', 'Avg Daily Turnover (B VND)', 'Trading Days'])

    df_turnover = _safe_read_excel('sql/turnover.xlsx')

    return theme_config, df_is_quarterly, df_bs_quarterly, df_forecast, df_liquidity, df_turnover


SEGMENTS = [
    {
        "key": "brokerage_fee",
        "label": "Brokerage Fee",
        "forecast_key": "Net_Brokerage_Income",
        "columns": ['Net_Brokerage_Income'],
    },
    {
        "key": "margin_income",
        "label": "Margin Income",
        "forecast_key": "Net_Margin_lending_Income",
        "columns": ['Net_Margin_lending_Income', 'Net_Margin_Lending_Income'],
    },
    {
        "key": "investment_income",
        "label": "Investment Income",
        "forecast_key": "Net_Investment",
        "columns": ['Net_investment_income'],
    },
    {
        "key": "ib_income",
        "label": "IB Income",
        "forecast_key": "Net_IB_Income",
        "columns": ['Net_IB_Income'],
    },
    {
        "key": "sga",
        "label": "SG&A",
        "forecast_key": "SG_A",
        "columns": ['SG_A'],
    },
    {
        "key": "interest_expense",
        "label": "Interest Expense",
        "forecast_key": "Interest_Expense",
        "columns": ['Interest_Expense'],
    },
]


def _normalize_name(name: str) -> str:
    return str(name).replace('_', '').replace('-', '').lower().strip()


def sum_columns(df: pd.DataFrame, columns):
    values = pd.Series(0.0, index=df.index, dtype=float)
    if isinstance(columns, str):
        columns = [columns]

    available_columns = {col: _normalize_name(col) for col in df.columns}

    for col in columns:
        target = _normalize_name(col)
        matches = [name for name, norm in available_columns.items() if norm == target]
        for match in matches:
            values = values.add(pd.to_numeric(df[match], errors='coerce').fillna(0.0), fill_value=0.0)

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
        df[segment['key']] = sum_columns(df, segment.get('columns', []))

    if 'PBT' in df.columns:
        df['pbt'] = pd.to_numeric(df['PBT'], errors='coerce').fillna(0.0)
    else:
        df['pbt'] = 0.0

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


def estimate_trading_days(lookup: dict, target_year: int, target_quarter: int) -> int:
    stats = lookup.get((target_year, target_quarter))
    if stats:
        td = stats.get('trading_days')
        if td and td >= 50:
            return int(td)

    historical_days = [
        stats.get('trading_days')
        for (year, quarter), stats in lookup.items()
        if quarter == target_quarter and stats.get('trading_days')
    ]

    filtered_days = [int(d) for d in historical_days if d and d >= 50]
    if filtered_days:
        return int(round(sum(filtered_days) / len(filtered_days)))

    return 66


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


@st.cache_data(ttl=3600)
def load_prop_book_data() -> pd.DataFrame:
    try:
        return pd.read_excel('sql/Prop book.xlsx')
    except Exception:
        return pd.DataFrame()


def sort_quarters_by_date(quarters: list[str]) -> list[str]:
    def key(q: str):
        try:
            q = q.strip()
            quarter_part = int(q[0])
            year_part = int(q[2:])
            full_year = 2000 + year_part if year_part < 50 else 1900 + year_part
            return full_year * 10 + quarter_part
        except Exception:
            return q

    return sorted(quarters, key=key)


def is_valid_ticker(ticker: str) -> bool:
    if not ticker:
        return False

    ticker_upper = ticker.upper().strip()
    invalid_patterns = [
        'OTHER', 'OTHERS', 'UNLISTED', 'PBT', 'TOTAL', 'CASH', 'BOND',
        'DEPOSIT', 'RECEIVABLE', 'PAYABLE', 'EQUITY', 'LIABILITY'
    ]

    if any(pattern in ticker_upper for pattern in invalid_patterns):
        return False

    if ' ' in ticker_upper:
        return False

    return bool(re.match(r'^[A-Z]{2,5}$', ticker_upper))


def fetch_historical_price(ticker: str) -> pd.DataFrame:
    if not is_valid_ticker(ticker):
        return pd.DataFrame()

    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": ticker.strip().upper(),
        "type": "stock",
        "resolution": "D",
        "from": "0",
        "to": str(int(datetime.now().timestamp()))
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'data' not in data or not data['data']:
            return pd.DataFrame()
        df = pd.DataFrame(data['data'])
        if 'tradingDate' in df.columns:
            if df['tradingDate'].dtype == 'object' and df['tradingDate'].astype(str).str.contains('T').any():
                df['tradingDate'] = pd.to_datetime(df['tradingDate'])
            else:
                df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms')
        return df
    except requests.exceptions.RequestException:
        return pd.DataFrame()


def get_close_price(df: pd.DataFrame, target_date: str | None = None):
    if df.empty:
        return None

    if target_date:
        target = pd.to_datetime(target_date)
        if target.tzinfo is None:
            target = target.tz_localize('UTC')
        subset = df[df['tradingDate'] <= target]
        if subset.empty:
            return None
        return subset.iloc[-1]['close']

    return df.iloc[-1]['close']


def get_quarter_end_prices(tickers: list[str], quarter: str) -> dict[str, float | None]:
    quarter_map = {"1Q": "-03-31", "2Q": "-06-30", "3Q": "-09-30", "4Q": "-12-31"}
    q_part, y_part = quarter[:2], quarter[2:]
    date_suffix = quarter_map.get(q_part)
    if not date_suffix:
        return {ticker: None for ticker in tickers}

    full_year = 2000 + int(y_part) if int(y_part) < 50 else 1900 + int(y_part)
    date_str = f"{full_year}{date_suffix}"

    prices: dict[str, float | None] = {}
    for ticker in tickers:
        if not is_valid_ticker(ticker):
            prices[ticker] = None
            continue

        cache_key = f"{ticker}_{quarter}_quarter_end"
        if cache_key in st.session_state.price_cache:
            prices[ticker] = st.session_state.price_cache[cache_key]
            continue

        price_df = fetch_historical_price(ticker)
        price = get_close_price(price_df, date_str)
        st.session_state.price_cache[cache_key] = price
        prices[ticker] = price

    return prices


def get_current_prices(tickers: list[str]) -> dict[str, float | None]:
    prices: dict[str, float | None] = {}
    for ticker in tickers:
        if not is_valid_ticker(ticker):
            prices[ticker] = None
            continue

        cache_key = f"{ticker}_current"
        if cache_key in st.session_state.price_cache:
            prices[ticker] = st.session_state.price_cache[cache_key]
            continue

        price_df = fetch_historical_price(ticker)
        price = get_close_price(price_df)
        st.session_state.price_cache[cache_key] = price
        prices[ticker] = price

    return prices


@st.cache_data(ttl=3600)
def load_investment_metrics(ticker: str, start_year: int) -> pd.DataFrame:
    try:
        return load_brokerage_metrics(ticker=ticker, start_year=start_year, include_annual=True)
    except Exception:
        return pd.DataFrame()


def calculate_fvtpl_profit_total(broker: str) -> tuple[float | None, str | None]:
    df_book = load_prop_book_data()
    if df_book.empty or 'FVTPL value' not in df_book.columns:
        return None, None

    mask = (
        (df_book['Broker'] == broker)
        & df_book['FVTPL value'].notnull()
        & (df_book['FVTPL value'] != 0)
    )
    broker_df = df_book[mask].copy()

    if broker_df.empty:
        return None, None

    quarters = sort_quarters_by_date(broker_df['Quarter'].unique().tolist())
    if not quarters:
        return None, None

    selected_quarter = None
    quarter_holdings: pd.DataFrame | None = None
    exclude_set = {'PBT', 'OTHERS'}

    for quarter_key in reversed(quarters):
        subset = broker_df[broker_df['Quarter'] == quarter_key].copy()
        subset['Ticker'] = subset['Ticker'].fillna('').astype(str)
        subset = subset[~subset['Ticker'].str.upper().isin(exclude_set)]
        if subset.empty:
            continue
        grouped = subset.groupby('Ticker', as_index=False)['FVTPL value'].sum()
        if grouped['FVTPL value'].abs().sum() == 0:
            continue
        selected_quarter = quarter_key
        quarter_holdings = grouped
        break

    if selected_quarter is None or quarter_holdings is None or quarter_holdings.empty:
        return None, None

    tickers = quarter_holdings['Ticker'].tolist()
    quarter_prices = get_quarter_end_prices(tickers, selected_quarter)
    current_prices = get_current_prices(tickers)

    profit_total = 0.0

    for _, row in quarter_holdings.iterrows():
        ticker = row['Ticker']
        value = float(row['FVTPL value'])
        quarter_price = quarter_prices.get(ticker)
        current_price = current_prices.get(ticker)

        if quarter_price in (None, 0) or current_price in (None, 0) or value == 0:
            continue

        volume = value / quarter_price
        profit = volume * (current_price - quarter_price)
        profit_total += profit

    if profit_total == 0.0:
        return None, None

    return profit_total, selected_quarter



theme_config, df_is_quarterly, df_bs_quarterly, df_forecast, df_liquidity_raw, df_turnover = load_data()

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
show_share_debug = st.sidebar.checkbox("Show market share debug", value=False)

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
df_liquidity = df_liquidity_raw.copy()

if not df_liquidity.empty:
    required_cols = {'Year', 'Quarter', 'Avg Daily Turnover (B VND)', 'Trading Days'}
    if not required_cols.issubset(df_liquidity.columns):
        df_liquidity = pd.DataFrame(columns=['Year', 'Quarter', 'Avg Daily Turnover (B VND)', 'Trading Days'])
else:
    df_liquidity = pd.DataFrame(columns=['Year', 'Quarter', 'Avg Daily Turnover (B VND)', 'Trading Days'])

def _compute_total_turnover(row: pd.Series) -> float | None:
    if pd.isna(row.get('Avg Daily Turnover (B VND)')) or pd.isna(row.get('Trading Days')):
        return None
    return float(row['Avg Daily Turnover (B VND)']) * float(row['Trading Days']) * 1e9

quarter_stats_lookup = {}
for _, liquidity_row in df_liquidity.iterrows():
    year = int(liquidity_row['Year'])
    quarter = int(liquidity_row['Quarter'])
    avg_daily_bn = float(liquidity_row['Avg Daily Turnover (B VND)']) if pd.notna(liquidity_row['Avg Daily Turnover (B VND)']) else None
    trading_days = int(liquidity_row['Trading Days']) if pd.notna(liquidity_row['Trading Days']) else None
    total_turnover = _compute_total_turnover(liquidity_row)

    quarter_stats_lookup[(year, quarter)] = {
        'total_turnover': total_turnover,
        'trading_days': trading_days,
        'avg_daily_bn': avg_daily_bn,
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
def fetch_market_share_quarter(year: int, quarter: int):
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
        records = []
        if data.get('success') and 'data' in data:
            for item in data['data'].get('brokerageStock', []):
                try:
                    percent = float(item.get('percentage', 0))
                except (TypeError, ValueError):
                    percent = None
                records.append({
                    'Brokerage_Code': item.get('shortenName', '').strip().upper(),
                    'Brokerage_Name': item.get('name', ''),
                    'Market_Share_Percent': percent,
                    'Year': year,
                    'Quarter': quarter,
                })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()

history_metrics = []
year_fee_observations = {}
broker_code_norm = get_brokerage_code(selected_broker)
broker_code_norm = broker_code_norm.strip().upper() if broker_code_norm else None
market_share_debug = []

for y, q in brokerage_history_quarters:
    stats = quarter_stats_lookup.get((y, q), {})
    total_turnover = stats.get('total_turnover')
    trading_days = stats.get('trading_days')
    avg_daily_bn = stats.get('avg_daily_bn')

    quarter_row = df_quarters[
        (df_quarters['YEARREPORT'] == y) & (df_quarters['LENGTHREPORT'] == q)
    ]
    net_brokerage = float(quarter_row['brokerage_fee'].iloc[0]) if not quarter_row.empty else None

    api_share_pct = None
    share_df = pd.DataFrame()
    if broker_code_norm:
        share_df = fetch_market_share_quarter(y, q)
        if not share_df.empty:
            match = share_df[share_df['Brokerage_Code'] == broker_code_norm]
            if not match.empty:
                api_share_pct = match.iloc[0]['Market_Share_Percent']

    share_decimal = None
    share_pct_display = None
    if api_share_pct:
        share_decimal = api_share_pct / 100
        share_pct_display = api_share_pct
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

    market_share_debug.append({
        'Period': quarter_label(y, q),
        'Broker Code': broker_code_norm or '-',
        'API Rows': int(len(share_df)) if not share_df.empty else 0,
        'API Share %': api_share_pct if api_share_pct is not None else None,
        'Turnover Share %': share_decimal * 100 if (share_decimal and api_share_pct is None) else None,
        'Final Share %': share_pct_display,
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

avg_daily_default = history_avg_daily[-1] if history_avg_daily else 0.0
if fee_decimal_default:
    fee_default = fee_decimal_default * 10000
elif history_fee_bps:
    fee_default = history_fee_bps[-1]
else:
    fee_default = 2.0

trading_days_forecast = estimate_trading_days(quarter_stats_lookup, target_year, target_quarter)

share_default_pct = share_default * 100 if share_default is not None else None

target_stats = quarter_stats_lookup.get((target_year, target_quarter))
if target_stats and target_stats.get('avg_daily_bn') is not None:
    avg_daily_default = target_stats['avg_daily_bn']

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
    'share_pct': market_share_input if (share_default_pct is not None or market_share_input > 0) else None,
    'fee_bps': net_fee_input,
    'net_brokerage_bn': net_brokerage_forecast_bn,
}

target_share_df = fetch_market_share_quarter(target_year, target_quarter) if broker_code_norm else pd.DataFrame()
target_api_share = None
if broker_code_norm and not target_share_df.empty:
    match = target_share_df[target_share_df['Brokerage_Code'] == broker_code_norm]
    if not match.empty:
        target_api_share = match.iloc[0]['Market_Share_Percent']
market_share_debug.append({
    'Period': target_label,
    'Broker Code': broker_code_norm or '-',
    'API Rows': int(len(target_share_df)) if not target_share_df.empty else 0,
    'API Share %': target_api_share,
    'Turnover Share %': share_default_pct if (share_default_pct is not None and target_api_share is None) else None,
    'Final Share %': forecast_metrics['share_pct'],
})

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

if show_share_debug:
    debug_df = pd.DataFrame(market_share_debug)
    st.markdown("#### Market Share Debug")
    st.dataframe(debug_df, use_container_width=True)
st.caption(f"Assuming {trading_days_forecast} trading days for {target_label} and applying net brokerage formula.")

def extract_is_value(year: int, quarter: int, codes: list[str]) -> float | None:
    subset = df_is_quarterly[
        (df_is_quarterly['TICKER'] == selected_broker)
        & (df_is_quarterly['YEARREPORT'] == year)
        & (df_is_quarterly['LENGTHREPORT'] == quarter)
    ]
    if subset.empty:
        return None

    row = subset.iloc[0]
    normalized_map = { _normalize_name(col): col for col in row.index }
    total = 0.0
    found = False
    for code in codes:
        norm = _normalize_name(code)
        match = normalized_map.get(norm)
        if match and match in row:
            value = pd.to_numeric(row[match], errors='coerce')
            if pd.notnull(value):
                total += float(value)
                found = True

    return total if found else None


def extract_bs_value(year: int, quarter: int, codes: list[str]) -> float | None:
    subset = df_bs_quarterly[
        (df_bs_quarterly['TICKER'] == selected_broker)
        & (df_bs_quarterly['YEARREPORT'] == year)
        & (df_bs_quarterly['LENGTHREPORT'] == quarter)
    ]
    if subset.empty:
        return None

    row = subset.iloc[0]
    total = 0.0
    found = False
    for code in codes:
        if code in row:
            value = pd.to_numeric(row[code], errors='coerce')
            if pd.notnull(value):
                total += float(value)
                found = True

    return total if found else None


MARGIN_INCOME_CODES = ['Net_Margin_lending_Income', 'IS.7', 'IS.30']
INTEREST_EXPENSE_CODES = ['IS.51']
MARGIN_BALANCE_CODES = ['BS.8']
BORROWING_BALANCE_CODES = ['BS.95', 'BS.100', 'BS.122', 'BS.127']

margin_history_metrics: list[dict[str, float | None]] = []

for y, q in brokerage_history_quarters:
    margin_balance_val = extract_bs_value(y, q, MARGIN_BALANCE_CODES)
    borrowing_balance_val = extract_bs_value(y, q, BORROWING_BALANCE_CODES)
    margin_income_val = extract_is_value(y, q, MARGIN_INCOME_CODES)
    interest_expense_val = extract_is_value(y, q, INTEREST_EXPENSE_CODES)

    margin_balance_bn = margin_balance_val / 1e9 if margin_balance_val is not None else None
    borrowing_balance_bn = borrowing_balance_val / 1e9 if borrowing_balance_val is not None else None

    margin_income_bn = abs(margin_income_val) / 1e9 if margin_income_val is not None else None
    interest_expense_bn = abs(interest_expense_val) / 1e9 if interest_expense_val is not None else None

    margin_rate_pct = None
    if margin_balance_val not in (None, 0) and margin_income_val is not None:
        margin_rate_pct = (margin_income_val * 4 / margin_balance_val) * 100

    interest_rate_pct = None
    if borrowing_balance_val not in (None, 0) and interest_expense_val is not None:
        interest_rate_pct = (abs(interest_expense_val) * 4 / borrowing_balance_val) * 100

    margin_history_metrics.append({
        'label': quarter_label(y, q),
        'margin_balance_bn': margin_balance_bn,
        'margin_rate_pct': margin_rate_pct,
        'margin_income_bn': margin_income_bn,
        'borrowing_balance_bn': borrowing_balance_bn,
        'interest_rate_pct': interest_rate_pct,
        'interest_expense_bn': interest_expense_bn,
    })


def last_history_value(key: str, default: float = 0.0) -> float:
    for entry in reversed(margin_history_metrics):
        value = entry.get(key)
        if value is not None:
            return float(value)
    return default


margin_balance_default = last_history_value('margin_balance_bn')
margin_rate_default = last_history_value('margin_rate_pct')
borrowing_balance_default = last_history_value('borrowing_balance_bn')
interest_rate_default = last_history_value('interest_rate_pct')

st.markdown("#### Margin Lending Forecast")
margin_cols = st.columns(3)
margin_balance_input = margin_cols[0].number_input(
    f"{target_label} Margin Balance (bn VND)",
    value=float(round(margin_balance_default, 0)),
    min_value=0.0,
    step=100.0,
    format="%.0f",
    key="margin_balance_input",
)

margin_rate_input = margin_cols[1].number_input(
    f"{target_label} Margin Lending Rate (%)",
    value=float(round(margin_rate_default, 2)),
    min_value=0.0,
    step=0.1,
    format="%.2f",
    key="margin_rate_input",
)

interest_rate_input = margin_cols[2].number_input(
    f"{target_label} Borrowing Rate (%)",
    value=float(round(interest_rate_default, 2)),
    min_value=0.0,
    step=0.1,
    format="%.2f",
    key="borrowing_rate_input",
)

margin_balance_base = margin_balance_default
borrowing_balance_base = borrowing_balance_default
interest_rate_assumed = interest_rate_input

delta_margin_balance = margin_balance_input - margin_balance_base
borrowing_balance_adjusted = max(borrowing_balance_base + delta_margin_balance, 0.0)

margin_income_forecast_bn = margin_balance_input * (margin_rate_input / 100) / 4 if margin_rate_input else 0.0
interest_expense_forecast_bn = (
    borrowing_balance_adjusted * (interest_rate_assumed / 100) / 4 if interest_rate_assumed else 0.0
)

margin_forecast_metrics = {
    'label': target_label,
    'margin_balance_bn': margin_balance_input,
    'margin_rate_pct': margin_rate_input,
    'margin_income_bn': margin_income_forecast_bn,
    'borrowing_balance_bn': borrowing_balance_adjusted,
    'interest_rate_pct': interest_rate_assumed,
    'interest_expense_bn': interest_expense_forecast_bn,
}

margin_table_rows = {
    'Metric': [
        'Margin Balance (bn)',
        'Margin Lending Rate (%)',
        'Margin Lending Income (bn)',
        'Borrowing Balance (bn)',
        'Interest Rate (%)',
        'Interest Expense (bn)',
    ]
}

for metrics in margin_history_metrics:
    margin_table_rows[metrics['label']] = [
        fmt_value(metrics['margin_balance_bn']),
        fmt_value(metrics['margin_rate_pct'], 2),
        fmt_value(metrics['margin_income_bn']),
        fmt_value(metrics['borrowing_balance_bn']),
        fmt_value(metrics['interest_rate_pct'], 2),
        fmt_value(metrics['interest_expense_bn']),
    ]

margin_table_rows[margin_forecast_metrics['label']] = [
    fmt_value(margin_forecast_metrics['margin_balance_bn']),
    fmt_value(margin_forecast_metrics['margin_rate_pct'], 2),
    fmt_value(margin_forecast_metrics['margin_income_bn']),
    fmt_value(margin_forecast_metrics['borrowing_balance_bn']),
    fmt_value(margin_forecast_metrics['interest_rate_pct'], 2),
    fmt_value(margin_forecast_metrics['interest_expense_bn']),
]

margin_table_df = pd.DataFrame(margin_table_rows)
margin_table_df = margin_table_df.set_index('Metric')
st.dataframe(margin_table_df, use_container_width=True)

st.caption("Margin lending income and interest expense calculated as balance × rate ÷ 4 for the forecast quarter.")


def render_segment_override(segment_key: str, title: str, input_key: str) -> tuple[float, float]:
    st.markdown(f"#### {title}")

    history = (
        df_actual[['YEARREPORT', 'LENGTHREPORT', segment_key]]
        .dropna(subset=[segment_key])
        .sort_values(['YEARREPORT', 'LENGTHREPORT'])
        .tail(4)
    )

    if not history.empty:
        history_display = history.copy()
        history_display['Quarter'] = history_display.apply(
            lambda row: quarter_label(int(row['YEARREPORT']), int(row['LENGTHREPORT'])),
            axis=1
        )
        history_display[f"{title} (bn)"] = history_display[segment_key].apply(lambda v: v / 1e9)
        display_df = history_display[['Quarter', f"{title} (bn)"]].copy()
        display_df[f"{title} (bn)"] = display_df[f"{title} (bn)"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No historical data available for {title.lower()}.")

    base_value = base_segments.get(segment_key, 0.0)
    base_bn = format_bn(base_value)
    if not math.isfinite(base_bn):
        base_bn = 0.0

    input_bn = st.number_input(
        f"{target_label} {title} (bn VND)",
        value=float(round(base_bn)),
        step=10.0,
        format="%.0f",
        key=input_key,
    )

    return float(input_bn), float(input_bn) * 1e9


ib_income_forecast_bn, ib_income_forecast_vnd = render_segment_override(
    'ib_income',
    'IB Income',
    'ib_income_input_override',
)

sga_forecast_bn, sga_forecast_vnd = render_segment_override(
    'sga',
    'SG&A Expense',
    'sga_expense_input_override',
)

st.markdown("#### Investment Book Snapshot")

investment_start_year = max(target_year - 3, 2017)
investment_metrics_df = load_investment_metrics(selected_broker, investment_start_year)

def format_investment_cell(value: float | None) -> str:
    if value is None or value == 0:
        return '-'
    return f"{value / 1e9:,.1f}"

if investment_metrics_df.empty:
    st.info("No investment holdings data available for the selected broker.")
else:
    quarterly_metrics = investment_metrics_df[investment_metrics_df['LENGTHREPORT'].between(1, 4)]

    if quarterly_metrics.empty:
        st.info("Investment book requires quarterly data. No quarterly records found.")
    else:
        period_records = (
            quarterly_metrics[['YEARREPORT', 'LENGTHREPORT']]
            .drop_duplicates()
            .sort_values(['YEARREPORT', 'LENGTHREPORT'], ascending=[False, False])
        )

        display_periods = period_records.head(6).sort_values(['YEARREPORT', 'LENGTHREPORT'])

        investment_rows: list[dict[str, str]] = []
        column_labels: list[tuple[int, int, str]] = []

        for _, record in display_periods.iterrows():
            year = int(record['YEARREPORT'])
            quarter = int(record['LENGTHREPORT'])
            column_labels.append((year, quarter, quarter_label(year, quarter)))

        all_items: set[tuple[str, str]] = set()
        period_data_cache: dict[tuple[int, int], dict] = {}

        for year, quarter, _ in column_labels:
            data = get_investment_data(investment_metrics_df, selected_broker, year, quarter)
            period_data_cache[(year, quarter)] = data
            for category in ['FVTPL', 'AFS', 'HTM']:
                market_values = data.get(category, {}).get('Market Value', {})
                for item in market_values.keys():
                    all_items.add((category, item))

        if not all_items:
            st.info("No investment holdings found for the selected broker in the recent quarters.")
        else:
            for category in ['FVTPL', 'AFS', 'HTM']:
                category_items = sorted([item for cat, item in all_items if cat == category])
                if not category_items:
                    continue

                header_row = {'Item': category}
                for _, _, label in column_labels:
                    header_row[label] = ''
                investment_rows.append(header_row)

                for item in category_items:
                    row = {'Item': f'  {item}'}
                    has_data = False
                    for year, quarter, label in column_labels:
                        mv = (
                            period_data_cache[(year, quarter)]
                            .get(category, {})
                            .get('Market Value', {})
                            .get(item)
                        )
                        if mv not in (None, 0):
                            row[label] = format_investment_cell(mv)
                            has_data = True
                        else:
                            row[label] = '-'
                    if has_data:
                        investment_rows.append(row)

                total_row = {'Item': f'Total {category}'}
                for year, quarter, label in column_labels:
                    total_mv = sum(
                        period_data_cache[(year, quarter)]
                        .get(category, {})
                        .get('Market Value', {})
                        .values()
                    )
                    total_row[label] = format_investment_cell(total_mv)
                investment_rows.append(total_row)

                spacer_row = {'Item': ''}
                for _, _, label in column_labels:
                    spacer_row[label] = ''
                investment_rows.append(spacer_row)

            investment_table = pd.DataFrame(investment_rows)
            st.dataframe(investment_table, use_container_width=True, hide_index=True)

            csv_data = investment_table.to_csv(index=False)
            st.download_button(
                label="Download Investment Book",
                data=csv_data,
                file_name=f"investment_book_{selected_broker}.csv",
                mime="text/csv",
                key=f"download_investment_book_{selected_broker}"
            )

            with st.expander("Understanding Investment Categories"):
                st.markdown("""
                **FVTPL** (Fair Value Through Profit or Loss): Trading securities held for short-term profit

                **AFS** (Available-for-Sale): Long-term financial assets measured at fair value through OCI

                **HTM** (Held-to-Maturity): Fixed maturity investments measured at amortized cost

                **Values shown**: Market Value (bn VND) across the most recent quarters
                """)

st.markdown(f"#### {target_label} Investment Income Override")

fvtpl_profit_value, fvtpl_reference_quarter = calculate_fvtpl_profit_total(selected_broker)

investment_history = (
    df_actual[['YEARREPORT', 'LENGTHREPORT', 'investment_income']]
    .dropna(subset=['investment_income'])
    .sort_values(['YEARREPORT', 'LENGTHREPORT'])
)

display_columns = ['Quarter', 'Investment Income (bn)']
investment_display_df = pd.DataFrame(columns=display_columns)

recent_investment_history = investment_history.tail(4)
if not recent_investment_history.empty:
    recent_investment_history = recent_investment_history.assign(
        Quarter=recent_investment_history.apply(
            lambda row: quarter_label(int(row['YEARREPORT']), int(row['LENGTHREPORT'])), axis=1
        ),
        **{"Investment Income (bn)": recent_investment_history['investment_income'].apply(lambda v: v / 1e9)}
    )[['Quarter', 'Investment Income (bn)']]
    investment_display_df = pd.concat([investment_display_df, recent_investment_history], ignore_index=True)

if fvtpl_profit_value is not None:
    reference_label = fvtpl_reference_quarter or "latest quarter"
    fvtpl_row = pd.DataFrame([
        {
            'Quarter': f"FVTPL P/L since {reference_label}",
            'Investment Income (bn)': fvtpl_profit_value,
        }
    ])
    investment_display_df = pd.concat([investment_display_df, fvtpl_row], ignore_index=True)

if not investment_display_df.empty:
    display_df = investment_display_df.copy()
    if 'Investment Income (bn)' in display_df.columns:
        display_df['Investment Income (bn)'] = pd.to_numeric(
            display_df['Investment Income (bn)'], errors='coerce'
        )
        display_df['Investment Income (bn)'] = display_df['Investment Income (bn)'].apply(
            lambda x: "-" if pd.isna(x) else f"{x:,.0f}"
        )

    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No historical investment income data available for the last quarters.")

investment_base_value = base_segments.get('investment_income', 0.0)
investment_base_bn = format_bn(investment_base_value)
if not math.isfinite(investment_base_bn):
    investment_base_bn = 0.0

investment_income_input = st.number_input(
    f"{target_label} Investment Income (bn VND)",
    value=float(round(investment_base_bn)),
    step=10.0,
    format="%.0f",
    key="investment_income_input",
)

investment_income_forecast_bn = float(investment_income_input)

# Update summary with forecast brokerage and margin inputs
target_column_label = f"{target_label} Base (bn VND)"
if target_column_label in summary_df.columns:
    summary_df.loc[summary_df['Segment'] == 'Brokerage Fee', target_column_label] = format_bn_str(net_brokerage_forecast)
    summary_df.loc[summary_df['Segment'] == 'Margin Income', target_column_label] = format_bn_str(margin_income_forecast_bn * 1e9)
    summary_df.loc[summary_df['Segment'] == 'IB Income', target_column_label] = format_bn_str(ib_income_forecast_vnd)
    summary_df.loc[summary_df['Segment'] == 'SG&A', target_column_label] = format_bn_str(sga_forecast_vnd)
    summary_df.loc[summary_df['Segment'] == 'Investment Income', target_column_label] = format_bn_str(investment_income_forecast_bn * 1e9)
    summary_df.loc[summary_df['Segment'] == 'Interest Expense', target_column_label] = format_bn_str(-interest_expense_forecast_bn * 1e9)

st.subheader("Baseline Breakdown")
st.caption(f"Base assumptions derived from {target_year} full-year forecast minus actual results up to {latest_label}.")
st.dataframe(summary_df, hide_index=True)

st.subheader(f"{target_label} Segment Assumptions")

segment_inputs = {}
input_columns = st.columns(3)

for idx, segment in enumerate(SEGMENTS):
    if segment['key'] in ('brokerage_fee', 'margin_income', 'ib_income', 'sga', 'investment_income', 'interest_expense'):
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
segment_inputs['margin_income'] = margin_income_forecast_bn * 1e9
segment_inputs['ib_income'] = ib_income_forecast_vnd
segment_inputs['sga'] = sga_forecast_vnd
segment_inputs['investment_income'] = investment_income_forecast_bn * 1e9
segment_inputs['interest_expense'] = -interest_expense_forecast_bn * 1e9

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
