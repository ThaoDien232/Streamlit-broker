"""Display stored quarterly brokerage sector commentary."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
import streamlit as st

from utils.db import run_query

st.set_page_config(page_title="Quarterly Sector Commentary", page_icon="🧾", layout="wide")

SECTOR_TICKER = "Sector"
CACHE_TTL_SECONDS = 600


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


def format_generated_at(timestamp: pd.Timestamp | None) -> str:
    if isinstance(timestamp, pd.Timestamp) and not pd.isna(timestamp):
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return "Unknown"


def sanitize_commentary(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).replace("\r\n", "\n").strip()


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_sector_commentary() -> pd.DataFrame:
    query = (
        "SELECT TICKER, QUARTER, COMMENTARY, GENERATED_AT\n"
        "FROM dbo.Brokerage_Comments\n"
        "WHERE TICKER = :ticker\n"
        "  AND COMMENTARY IS NOT NULL\n"
        "  AND LTRIM(RTRIM(COMMENTARY)) <> ''"
    )

    df = run_query(query, {"ticker": SECTOR_TICKER})

    if df.empty:
        return df

    df["TICKER"] = df["TICKER"].astype(str).str.strip()
    df["QUARTER"] = df["QUARTER"].astype(str).str.strip()

    if "GENERATED_AT" in df.columns:
        df["GENERATED_AT"] = pd.to_datetime(df["GENERATED_AT"], errors="coerce")

    df = df.sort_values(["QUARTER", "GENERATED_AT"], ascending=[True, False])

    return df.reset_index(drop=True)


def main() -> None:
    commentary_df = load_sector_commentary()

    if commentary_df.empty:
        st.warning("No sector-level commentaries found in the database.")
        return

    st.title("🧾 Quarterly Brokerage Sector Commentary")
    st.caption("Review sector insights stored directly from the SQL database.")

    quarter_options = sort_quarters_desc(commentary_df["QUARTER"].dropna())

    if not quarter_options:
        st.warning("No valid quarter labels available for sector commentary.")
        return

    selected_quarter = st.selectbox("Quarter", quarter_options)

    quarter_rows = commentary_df[commentary_df["QUARTER"] == selected_quarter].copy()

    if quarter_rows.empty:
        st.info("No sector commentary found for the selected quarter.")
        return

    latest_row = quarter_rows.iloc[0]
    commentary_text = sanitize_commentary(latest_row.get("COMMENTARY"))
    generated_at = format_generated_at(latest_row.get("GENERATED_AT"))
    ticker = latest_row.get("TICKER", SECTOR_TICKER)

    st.caption(f"Ticker: {ticker} · Last updated: {generated_at}")

    if not commentary_text:
        st.info("Commentary text is missing for the selected quarter.")
    else:
        st.markdown(commentary_text)

    previous_rows = quarter_rows.iloc[1:]

    if not previous_rows.empty:
        with st.expander("Previous versions", expanded=False):
            for index, row in enumerate(previous_rows.itertuples(index=False), start=1):
                version_generated = format_generated_at(getattr(row, "GENERATED_AT", None))
                version_commentary = sanitize_commentary(getattr(row, "COMMENTARY", ""))
                st.markdown(f"**Version {index}** — Generated at: {version_generated}")
                if version_commentary:
                    st.markdown(version_commentary)
                else:
                    st.caption("No commentary text available.")
                st.divider()


if __name__ == "__main__":
    main()
