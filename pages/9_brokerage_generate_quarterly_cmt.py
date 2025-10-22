"""Generate aggregated quarterly commentary for brokerage sector."""

from __future__ import annotations

import re
import textwrap
from typing import Iterable

import pandas as pd
import streamlit as st

from utils.brokerage_codes import BROKERAGE_CODE_MAP
from utils.db import run_query
from utils.openai_commentary import get_openai_client


st.set_page_config(page_title="Generate Quarterly Commentary", page_icon="🧾", layout="wide")


DEFAULT_MODEL = "gpt-5"

SYSTEM_MESSAGE = (
    "You are an expert financial analyst covering Vietnamese brokerage firms. "
    "Synthesize sector insights using only the information provided in the user prompt."
)

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a senior analyst with expertise in the Vietnamese brokerage sector. Please analyze the following {bank_count} brokerage comments for {quarter} and provide a comprehensive analysis.

    BROKERAGE COMMENTS FOR {quarter}:
    {comments_text}

    Please provide analysis in the following three sections:

    ## 1. KEY CHANGES SUMMARY
    Summarize the most significant trends and changes across all brokers in this quarter. Focus on:
    - Overall brokerage sector performance and market conditions
    - Common themes and patterns across brokers, highlighting shared improvements and concerns
    - Key drivers impacting multiple brokers

    ## 2. SENTIMENT ANALYSIS & NOTABLE BROKERS
    Analyze the tone and sentiment of comments:
    - Overall sector sentiment (positive/neutral/negative)
    - Brokers with the most positive developments and the specific reasons why
    - Brokers with the most concerning issues and the specific reasons for concern

    ## 3. SIGNIFICANT BROKER CHANGES BY TOPIC
    Identify which specific brokers showed the most significant changes in each key area:

    **Traditional brokerage: Margin lending & Trading fee**
    - Most improved: [Specific Broker Name] - [detailed reason based on comment data]
    - Most concerning: [Specific Broker Name] - [detailed reason based on comment data]

    **Investment and Equity Portfolio**
    - Strongest growth: [Specific Broker Name] - [detailed reason based on comment data]
    - Weakest/declining: [Specific Broker Name] - [detailed reason based on comment data]

    **Instructions:**
    - Be specific with actual broker names (tickers) mentioned in the comments
    - Write in bullet points format, keeping the tone punchy and concise
    - Provide clear, data-driven reasoning based on the comments provided
    - Use quantitative insights where available in the comments
    - Maintain a professional analyst tone
    - If insufficient data for a category, clearly state "Insufficient data available"
    """
)


def get_display_ticker(data_ticker: str) -> str:
    """Map an internal data ticker (e.g. TCBS) to its display ticker (e.g. TCX)."""

    if not data_ticker:
        return data_ticker

    for display_code, data_code in BROKERAGE_CODE_MAP.items():
        if data_code.upper() == str(data_ticker).upper():
            return display_code

    return str(data_ticker).upper()


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


@st.cache_data(ttl=600)
def load_commentary_data() -> pd.DataFrame:
    query = textwrap.dedent(
        """
        SELECT TICKER, QUARTER, COMMENTARY, GENERATED_AT
        FROM dbo.Brokerage_Comments
        WHERE COMMENTARY IS NOT NULL AND LTRIM(RTRIM(COMMENTARY)) <> ''
        """
    )

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


def format_multiline_block(text: str) -> str:
    sanitized = str(text or "").replace("\r\n", "\n").strip()
    if not sanitized:
        return "    No commentary provided.\n"

    lines = sanitized.split("\n")
    return "".join(f"    {line.rstrip()}\n" for line in lines)


def build_prompt(quarter: str, rows: pd.DataFrame) -> tuple[str, int]:
    comment_blocks: list[str] = []

    for row in rows.itertuples(index=False):
        comment = getattr(row, "COMMENTARY", "")
        sanitized = str(comment or "").strip()
        if not sanitized:
            continue

        display_ticker = get_display_ticker(getattr(row, "TICKER", ""))
        block_content = format_multiline_block(sanitized)
        block = (
            f"- ticker: {display_ticker}\n"
            f"  bank_comment: |\n"
            f"{block_content}"
            f"  comment_text: |\n"
            f"{block_content}"
        ).rstrip()
        comment_blocks.append(block)

    bank_count = len(comment_blocks)
    comments_text = "\n".join(comment_blocks)

    prompt = PROMPT_TEMPLATE.format(
        bank_count=bank_count,
        quarter=quarter,
        comments_text=comments_text if comments_text else "No brokerage comments available."
    )

    return prompt, bank_count


def _extract_message_content(message) -> str:
    if message is None:
        return ""

    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content")
                if isinstance(text_value, str):
                    parts.append(text_value)
            else:
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part).strip()

    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str):
        return text_attr.strip()

    return ""


def call_openai(prompt: str, model: str) -> str:
    client = get_openai_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_completion_tokens=2000,
    )

    message = response.choices[0].message
    content = _extract_message_content(message)

    if not content:
        raise ValueError("OpenAI returned an empty commentary for the generated prompt.")

    return content


RESULT_KEY = "quarterly_sector_commentary"
PROMPT_KEY = "quarterly_sector_prompt"
META_KEY = "quarterly_sector_metadata"
QUARTER_KEY = "quarterly_sector_selected_quarter"


def show_commentary_result(header: str, commentary: str, metadata: str | None = None) -> None:
    st.markdown("### " + header)
    if metadata:
        st.caption(metadata)
    st.markdown(commentary)


def main() -> None:
    commentary_df = load_commentary_data()

    if commentary_df.empty:
        st.warning("No broker commentaries found in the database.")
        return

    quarter_options = sort_quarters_desc(commentary_df["QUARTER"].dropna())

    if not quarter_options:
        st.warning("No valid quarter labels available in the commentary dataset.")
        return

    st.title("🧾 Generate Quarterly Brokerage Commentary")
    st.caption("Aggregate individual broker insights into a sector-wide quarterly narrative.")

    selected_quarter = st.selectbox("Quarter", quarter_options)

    if st.session_state.get(QUARTER_KEY) != selected_quarter:
        st.session_state.pop(RESULT_KEY, None)
        st.session_state.pop(PROMPT_KEY, None)
        st.session_state[QUARTER_KEY] = selected_quarter

    quarter_rows = commentary_df[commentary_df["QUARTER"] == selected_quarter].copy()

    if quarter_rows.empty:
        st.info("No broker commentaries found for the selected quarter.")
        return

    quarter_rows["DisplayTicker"] = quarter_rows["TICKER"].map(get_display_ticker)
    quarter_rows["GeneratedAt"] = quarter_rows["GENERATED_AT"].map(format_generated_at)

    prompt, bank_count = build_prompt(selected_quarter, quarter_rows)

    if bank_count == 0:
        st.info("Commentary text is missing for all brokers in this quarter.")
        return

    st.metric(label="Brokers included", value=bank_count)

    with st.expander("Broker comment samples", expanded=False):
        preview_table = quarter_rows[["DisplayTicker", "COMMENTARY", "GeneratedAt"]].rename(
            columns={"DisplayTicker": "Broker", "COMMENTARY": "Commentary", "GeneratedAt": "Generated"}
        )
        st.dataframe(preview_table, hide_index=True, use_container_width=True)

    with st.expander("Prompt preview", expanded=False):
        st.code(prompt, language="markdown")

    model = st.selectbox("OpenAI model", (DEFAULT_MODEL, "gpt-4", "gpt-4o", "gpt-4o-mini"))

    generate_button = st.button("Generate quarterly commentary", type="primary")

    if generate_button:
        with st.spinner("Calling OpenAI to generate commentary..."):
            try:
                commentary = call_openai(prompt, model)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to generate commentary: {exc}")
                return
        commentary = commentary.strip()
        if not commentary:
            st.warning("Generation completed but produced no content. Please try again or adjust the prompt.")
            return

        metadata = f"Generated with model: {model}"

        st.session_state[RESULT_KEY] = commentary
        st.session_state[PROMPT_KEY] = prompt
        st.session_state[META_KEY] = metadata
        st.success("Quarterly commentary generated.")
        show_commentary_result("Quarterly Commentary", commentary, metadata)
        with st.expander("Prompt used", expanded=False):
            st.code(prompt, language="markdown")
        return

    generated_commentary = st.session_state.get(RESULT_KEY)
    if generated_commentary:
        metadata = st.session_state.get(META_KEY, "Generated just now")
        saved_prompt = st.session_state.get(PROMPT_KEY)
        show_commentary_result("Quarterly Commentary", generated_commentary, metadata)

        if saved_prompt:
            with st.expander("Prompt used", expanded=False):
                st.code(saved_prompt, language="markdown")


if __name__ == "__main__":
    main()
