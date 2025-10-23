"""
Stock price fetching utilities for prop holdings volume calculations.
Lightweight functions focused on getting quarter-end closing prices only.
"""

import pandas as pd
import requests
import streamlit as st
from datetime import datetime


@st.cache_data(ttl=86400)  # Cache for 24 hours - historical prices don't change
def fetch_close_prices(ticker: str) -> pd.DataFrame:
    """
    Fetch daily closing prices from TCBS API for the given ticker.
    Returns DataFrame with 'tradingDate' and 'close' columns only.

    Cached for 24 hours since historical price data is immutable.
    """
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

        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])

            # Convert tradingDate to datetime
            if 'tradingDate' in df.columns:
                if df['tradingDate'].dtype == 'object' and df['tradingDate'].str.contains('T').any():
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'])
                else:
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms')

            # Only keep tradingDate and close price
            if 'tradingDate' in df.columns and 'close' in df.columns:
                return df[['tradingDate', 'close']]
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_quarter_end_price(ticker: str, quarter_label: str) -> float:
    """
    Get quarter-end closing price for a single ticker.

    Args:
        ticker: Stock ticker (e.g., 'VNM', 'HPG')
        quarter_label: Quarter label (e.g., '1Q24', '2Q25')

    Returns:
        Quarter-end closing price, or None if not available

    Cached for 24 hours - quarter-end prices are historical and don't change.
    """
    # Map quarter to end date
    q_map = {"1Q": "-03-31", "2Q": "-06-30", "3Q": "-09-30", "4Q": "-12-31"}

    try:
        q_part = quarter_label[:2]  # e.g., "1Q"
        y_part = quarter_label[2:]  # e.g., "24"

        # Convert to full year
        year = 2000 + int(y_part) if int(y_part) < 50 else 1900 + int(y_part)

        # Build date string (e.g., "2024-03-31")
        date_str = f"{year}{q_map.get(q_part, '-12-31')}"
        target_date = pd.to_datetime(date_str).tz_localize('UTC')

        # Fetch closing prices
        price_df = fetch_close_prices(ticker)

        if price_df.empty:
            return None

        # Get price on or before target date
        df_filtered = price_df[price_df['tradingDate'] <= target_date]

        if df_filtered.empty:
            return None

        return df_filtered.iloc[-1]['close']

    except Exception:
        return None


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_quarter_end_prices_batch(ticker_list: list, quarter_label: str) -> dict:
    """
    Get quarter-end prices for multiple tickers at once.

    Args:
        ticker_list: List of stock tickers
        quarter_label: Quarter label (e.g., '1Q24', '2Q25')

    Returns:
        Dictionary mapping ticker to quarter-end price
        Example: {'VNM': 80000, 'HPG': 25000}

    Each ticker's price is cached individually via get_quarter_end_price().
    Subsequent calls with overlapping tickers will use the cache.
    """
    prices = {}
    for ticker in ticker_list:
        prices[ticker] = get_quarter_end_price(ticker, quarter_label)

    return prices
