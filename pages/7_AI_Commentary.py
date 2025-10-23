"""
AI Brokerage Commentary Viewer
Displays stored AI-generated broker commentaries from the database.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
import streamlit as st

from utils.db import run_query
from utils.brokerage_codes import BROKERAGE_CODE_MAP


st.set_page_config(page_title="AI Commentary", page_icon="🤖", layout="wide")


def get_display_ticker(data_ticker: str) -> str:
    """Map an internal data ticker (e.g. TCBS) to its display ticker (e.g. TCX)."""

    if not data_ticker:
        return data_ticker

    for display_code, data_code in BROKERAGE_CODE_MAP.items():
        if data_code.upper() == data_ticker.upper():
            return display_code

    return data_ticker.upper()


def _normalize_year(year_str: str) -> int | None:
    try:
        year = int(year_str)
    except (TypeError, ValueError):
        return None

    if len(str(year_str)) == 2:
        return 2000 + year if year < 70 else 1900 + year

    return year


def parse_quarter_label(label: str) -> tuple[int, int] | None:
    """Parse quarter strings like 1Q24, 2024-Q3, Q32024 into (year, quarter)."""

    if not label or (isinstance(label, float) and pd.isna(label)):
        return None

    text = str(label).strip().upper()
    patterns = [
        r"^(?P<q>[1-4])Q(?P<y>\d{2,4})$",
        r"^(?P<y>\d{4})-?Q(?P<q>[1-4])$",
        r"^Q(?P<q>[1-4])(?P<y>\d{2,4})$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            quarter = int(match.group("q"))
            year = _normalize_year(match.group("y"))
            if year is not None:
                return year, quarter

    return None


def sort_quarters_desc(labels: Iterable[str]) -> list[str]:
    def sort_key(value: str) -> tuple[int, int]:
        parsed = parse_quarter_label(value)
        return parsed if parsed else (0, 0)

    return sorted({lbl for lbl in labels if lbl}, key=sort_key, reverse=True)


@st.cache_data(ttl=600)
def load_commentary_data() -> pd.DataFrame:
    query = """
        SELECT TICKER, QUARTER, COMMENTARY, GENERATED_AT
        FROM dbo.Brokerage_Comments
        WHERE COMMENTARY IS NOT NULL AND LTRIM(RTRIM(COMMENTARY)) <> ''
    """

    df = run_query(query)

    if df.empty:
        return df

    df["TICKER"] = df["TICKER"].astype(str).str.strip()
    df["QUARTER"] = df["QUARTER"].astype(str).str.strip()

    if "GENERATED_AT" in df.columns:
        df["GENERATED_AT"] = pd.to_datetime(df["GENERATED_AT"], errors="coerce")

    df = df.sort_values(["TICKER", "QUARTER", "GENERATED_AT"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["TICKER", "QUARTER"], keep="first")

    return df.reset_index(drop=True)


def format_generated_at(timestamp: pd.Timestamp | None) -> str:
    if isinstance(timestamp, pd.Timestamp) and not pd.isna(timestamp):
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return "Unknown"


def main() -> None:
    st.title("🤖 AI Brokerage Commentary")
    st.caption("Read-only view of stored quarterly AI analyses from the data warehouse.")

    commentary_df = load_commentary_data()

    if commentary_df.empty:
        st.warning("No AI commentaries found in the database.")
        return

    available_tickers = commentary_df["TICKER"].dropna().unique()
    ticker_map = {get_display_ticker(ticker): ticker for ticker in available_tickers}
    display_tickers = sorted(ticker_map.keys())

    selected_display_ticker = st.selectbox("Broker", display_tickers)
    selected_data_ticker = ticker_map[selected_display_ticker]

    ticker_rows = commentary_df[commentary_df["TICKER"] == selected_data_ticker]

    if ticker_rows.empty:
        st.info("No commentary available for the selected broker.")
        return

    quarter_options = sort_quarters_desc(ticker_rows["QUARTER"].dropna())

    if not quarter_options:
        st.info("Commentary quarters could not be determined for this broker.")
        return

    selected_quarter = st.selectbox("Quarter", quarter_options)

    selected_row = ticker_rows[ticker_rows["QUARTER"] == selected_quarter].sort_values(
        "GENERATED_AT", ascending=False
    ).head(1)

    if selected_row.empty:
        st.info("No commentary found for the selected quarter.")
        return

    commentary_text = selected_row.iloc[0]["COMMENTARY"]
    generated_at = selected_row.iloc[0].get("GENERATED_AT")

    st.subheader(f"{selected_display_ticker} — {selected_quarter}")
    st.caption(f"Generated at: {format_generated_at(generated_at)}")

    st.markdown("### Commentary")
    formatted_commentary = str(commentary_text).replace("\r\n", "\n")
    formatted_commentary = formatted_commentary.replace("\n", "  \n")
    st.markdown(formatted_commentary)

    st.divider()

    history_table = ticker_rows.copy()
    history_table["DisplayTicker"] = history_table["TICKER"].map(get_display_ticker)
    history_table["GeneratedAt"] = history_table["GENERATED_AT"].map(format_generated_at)

    sort_keys = history_table["QUARTER"].map(parse_quarter_label)
    history_table["SortYear"] = sort_keys.map(lambda value: value[0] if value else -1)
    history_table["SortQuarter"] = sort_keys.map(lambda value: value[1] if value else -1)

    history_table = history_table.sort_values(
        ["SortYear", "SortQuarter", "GENERATED_AT"],
        ascending=[False, False, False],
    )

    history_table = history_table[["DisplayTicker", "QUARTER", "GeneratedAt"]].rename(
        columns={"QUARTER": "Quarter"}
    )

    st.markdown("### Available Commentaries")
    st.dataframe(history_table, hide_index=True)


if __name__ == "__main__":
    main()
