"""
OpenAI integration for generating quarterly business performance commentary.
Uses Combined_Financial_Data.csv and broker-specific analysis.
"""

import os
from datetime import datetime
from typing import Iterable

import openai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _normalize_container(container: object) -> dict[str, str]:
    if isinstance(container, dict):
        return container
    try:
        return dict(container)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        normalized: dict[str, str] = {}
        try:
            for key in container.keys():  # type: ignore[attr-defined]
                normalized[key] = container[key]
        except Exception:  # noqa: BLE001
            pass
        return normalized


def _lookup_secret(container: object, candidates: Iterable[str]) -> str | None:
    mapping = _normalize_container(container)
    for candidate in candidates:
        value = mapping.get(candidate)
        if value:
            return value
    lowered = {key.lower(): key for key in mapping.keys()}
    for candidate in candidates:
        lowered_candidate = candidate.lower()
        if lowered_candidate in lowered:
            value = mapping[lowered[lowered_candidate]]
            if value:
                return value
    return None


def get_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets or environment."""

    api_key = None

    try:
        if "openai" in st.secrets:
            api_key = _lookup_secret(st.secrets["openai"], ("api_key", "API_KEY", "key", "new_key"))
        if not api_key:
            api_key = _lookup_secret(st.secrets, ("new_key",))
    except Exception:  # noqa: BLE001
        api_key = None

    if not api_key:
        api_key = _lookup_secret(os.environ, ("new_key",))

    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please add it to Streamlit secrets (new_key) or set the new_key environment variable."
        )

    return openai.OpenAI(api_key=api_key)

def get_calc_metric_value(df: pd.DataFrame, ticker: str, year: int, quarter: int, metric_code: str) -> float:
    """Get a specific calculated metric value from CALC statement type - same as Historical page"""
    result = df[
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter) &
        (df['STATEMENT_TYPE'] == 'CALC') &
        (df['METRIC_CODE'] == metric_code)
    ]

    if len(result) > 0:
        return result.iloc[0]['VALUE']
    return 0

def parse_quarter_label(quarter_label: str) -> tuple:
    """Parse quarter label like '1Q24' to (year, quarter_num)"""
    try:
        if not isinstance(quarter_label, str) or 'Q' not in quarter_label:
            return (None, None)

        quarter_label = quarter_label.strip()
        parts = quarter_label.split('Q')

        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            return (None, None)

        quarter_num = int(parts[0])
        year_str = parts[1]

        # Handle 2-digit years (assume 2000s if <= 50, otherwise 1900s)
        if len(year_str) == 2:
            year_int = int(year_str)
            if year_int <= 50:  # 00-50 = 2000-2050
                year = 2000 + year_int
            else:  # 51-99 = 1951-1999
                year = 1900 + year_int
        else:
            year = int(year_str)

        # Validate reasonable ranges
        if quarter_num not in [1, 2, 3, 4] or year < 1990 or year > 2050:
            return (None, None)

        return (year, quarter_num)
    except (ValueError, IndexError, AttributeError):
        return (None, None)

def get_broker_data(ticker: str, year_quarter: str, df: pd.DataFrame) -> dict:
    """
    Extract relevant financial data for a broker in a specific quarter.
    Uses the same approach as Historical page with CALC metrics.

    Args:
        ticker: Broker ticker (e.g., 'SSI', 'VCI')
        year_quarter: Quarter in format like '1Q24', '2Q23', etc.
        df: Combined_Financial_Data DataFrame

    Returns:
        Dictionary with financial metrics for the quarter
    """
    # Parse quarter label to get year and quarter number
    year, quarter_num = parse_quarter_label(year_quarter)

    if year is None or quarter_num is None:
        return None

    # Extract key financial metrics using the same approach as Historical page
    metrics = {
        'ticker': ticker,
        'quarter': year_quarter,
        'year': year,
        'quarter_num': quarter_num,
        # Revenue streams
        'total_operating_income': get_calc_metric_value(df, ticker, year, quarter_num, 'TOTAL_OPERATING_INCOME'),
        'net_brokerage_income': get_calc_metric_value(df, ticker, year, quarter_num, 'NET_BROKERAGE_INCOME'),
        'net_trading_income': get_calc_metric_value(df, ticker, year, quarter_num, 'NET_TRADING_INCOME'),
        'net_ib_income': get_calc_metric_value(df, ticker, year, quarter_num, 'NET_IB_INCOME'),
        'fee_income': get_calc_metric_value(df, ticker, year, quarter_num, 'FEE_INCOME'),
        'net_investment_income': get_calc_metric_value(df, ticker, year, quarter_num, 'NET_INVESTMENT_INCOME'),
        'margin_lending_income': get_calc_metric_value(df, ticker, year, quarter_num, 'MARGIN_LENDING_INCOME'),

        # Profitability
        'net_profit': get_calc_metric_value(df, ticker, year, quarter_num, 'NPAT'),
        'pbt': get_calc_metric_value(df, ticker, year, quarter_num, 'PBT'),
        'sga': get_calc_metric_value(df, ticker, year, quarter_num, 'SGA'),

        # Key ratios (convert from decimal to percentage)
        'roa': get_calc_metric_value(df, ticker, year, quarter_num, 'ROA') * 100,
        'roe': get_calc_metric_value(df, ticker, year, quarter_num, 'ROE') * 100,

        # Balance sheet
        'total_assets': get_calc_metric_value(df, ticker, year, quarter_num, 'TOTAL_ASSETS'),
        'total_equity': get_calc_metric_value(df, ticker, year, quarter_num, 'TOTAL_EQUITY'),
        'borrowing_balance': get_calc_metric_value(df, ticker, year, quarter_num, 'BORROWING_BALANCE'),
        'margin_balance': get_calc_metric_value(df, ticker, year, quarter_num, 'MARGIN_BALANCE'),
    }

    # Calculate revenue for compatibility (sum of main income streams)
    metrics['revenue'] = (
        metrics['total_operating_income'] if metrics['total_operating_income'] != 0
        else (metrics['net_brokerage_income'] + metrics['net_trading_income'] +
              metrics['net_ib_income'] + metrics['fee_income'])
    )

    return metrics

def get_last_6_quarters_data(ticker: str, current_quarter: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Get data for the last 6 quarters including the current quarter.

    Args:
        ticker: Broker ticker
        current_quarter: Current quarter like '1Q24'
        df: Combined_Financial_Data DataFrame

    Returns:
        DataFrame with last 6 quarters of data including absolute values and growth rates
    """
    # Get all available quarters for this ticker (sorted newest first)
    available_quarters = get_available_quarters(df, ticker)

    if not available_quarters or current_quarter not in available_quarters:
        return pd.DataFrame()

    # Find index of current quarter
    current_index = available_quarters.index(current_quarter)

    # Get last 6 quarters (including current)
    last_6_quarters = available_quarters[current_index:min(current_index + 6, len(available_quarters))]

    # Collect data for each quarter
    quarters_data = []
    for quarter in last_6_quarters:
        quarter_metrics = get_broker_data(ticker, quarter, df)
        if quarter_metrics:
            quarters_data.append(quarter_metrics)

    if not quarters_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df_quarters = pd.DataFrame(quarters_data)

    # Calculate growth rates (QoQ and YoY)
    metrics_to_track = ['total_operating_income', 'net_brokerage_income', 'net_investment_income',
                        'margin_lending_income', 'pbt', 'roa', 'roe', 'margin_balance']

    for metric in metrics_to_track:
        if metric in df_quarters.columns:
            # QoQ growth
            df_quarters[f'{metric}_qoq'] = df_quarters[metric].pct_change(periods=-1) * 100

            # YoY growth (4 quarters ago)
            if len(df_quarters) >= 5:
                df_quarters[f'{metric}_yoy'] = df_quarters[metric].pct_change(periods=-4) * 100

    return df_quarters

def get_comparative_data(ticker: str, current_quarter: str, df: pd.DataFrame) -> dict:
    """
    Get comparative data for previous quarters and year-over-year comparison.

    Args:
        ticker: Broker ticker
        current_quarter: Current quarter like '1Q24'
        df: Combined_Financial_Data DataFrame

    Returns:
        Dictionary with comparative metrics
    """
    # Get all available quarters for this ticker
    available_quarters = get_available_quarters(df, ticker)

    if len(available_quarters) < 2 or current_quarter not in available_quarters:
        return {}

    # Get current quarter data
    current_data = get_broker_data(ticker, current_quarter, df)
    if not current_data:
        return {}

    # Find previous quarter
    current_index = available_quarters.index(current_quarter)
    if current_index < len(available_quarters) - 1:
        previous_quarter = available_quarters[current_index + 1]
        previous_data = get_broker_data(ticker, previous_quarter, df)

        if previous_data:
            return {
                'previous_quarter': previous_quarter,
                'revenue_growth': ((current_data['revenue'] - previous_data['revenue']) / previous_data['revenue'] * 100) if previous_data['revenue'] != 0 else 0,
                'profit_growth': ((current_data['net_profit'] - previous_data['net_profit']) / previous_data['net_profit'] * 100) if previous_data['net_profit'] != 0 else 0,
                'roa_change': current_data['roa'] - previous_data['roa'],
                'roe_change': current_data['roe'] - previous_data['roe']
            }

    return {}

def format_financial_data(metrics: dict, comparative: dict = None) -> str:
    """
    Format financial data into a readable string for AI prompt.

    Args:
        metrics: Current quarter financial metrics
        comparative: Comparative data (optional)

    Returns:
        Formatted string with financial data
    """
    if not metrics:
        return "No financial data available for this period."

    # Format large numbers (in billions VND)
    def format_number(value):
        if abs(value) >= 1e12:
            return f"{value/1e12:.1f}T VND"
        elif abs(value) >= 1e9:
            return f"{value/1e9:.1f}B VND"
        elif abs(value) >= 1e6:
            return f"{value/1e6:.1f}M VND"
        else:
            return f"{value:.0f} VND"

    formatted_data = f"""
FINANCIAL PERFORMANCE - {metrics['ticker']} ({metrics['quarter']}):

Revenue Breakdown:
- Total Operating Income: {format_number(metrics['total_operating_income'])}
- Net Brokerage Income: {format_number(metrics['net_brokerage_income'])}
- Net Trading Income: {format_number(metrics['net_trading_income'])}
- Investment Banking Income: {format_number(metrics['net_ib_income'])}
- Fee Income: {format_number(metrics['fee_income'])}
- Net Investment Income: {format_number(metrics['net_investment_income'])}

Profitability:
- Net Profit After Tax (NPAT): {format_number(metrics['net_profit'])}
- Profit Before Tax (PBT): {format_number(metrics['pbt'])}
- Sales, General & Administrative (SGA): {format_number(metrics['sga'])}

Key Ratios:
- Return on Assets (ROA): {metrics['roa']:.2f}%
- Return on Equity (ROE): {metrics['roe']:.2f}%

Balance Sheet:
- Total Assets: {format_number(metrics['total_assets'])}
- Total Equity: {format_number(metrics['total_equity'])}
- Borrowing Balance: {format_number(metrics['borrowing_balance'])}
"""

    if comparative:
        formatted_data += f"""
COMPARATIVE ANALYSIS (vs {comparative.get('previous_quarter', 'Previous Quarter')}):
- Revenue Growth: {comparative.get('revenue_growth', 0):.1f}%
- Profit Growth: {comparative.get('profit_growth', 0):.1f}%
- ROA Change: {comparative.get('roa_change', 0):.2f}pp
- ROE Change: {comparative.get('roe_change', 0):.2f}pp
"""

    return formatted_data

def generate_commentary(ticker: str, year_quarter: str, df: pd.DataFrame,
                       model: str = "gpt-4", force_regenerate: bool = False,
                       analysis_table: pd.DataFrame = None,
                       market_share_table: pd.DataFrame = None,
                       prop_holdings_table: pd.DataFrame = None,
                       investment_composition_table: pd.DataFrame = None,
                       toi_drivers_qoq: pd.DataFrame = None,
                       toi_drivers_yoy: pd.DataFrame = None,
                       return_prompt: bool = False) -> str:
    """
    Generate AI commentary for a broker's quarterly performance using prepared analysis table.

    Args:
        ticker: Broker ticker
        year_quarter: Quarter (e.g., '1Q24')
        df: Combined_Financial_Data DataFrame
        model: OpenAI model to use
        force_regenerate: Whether to bypass cache
        analysis_table: Pre-built analysis table with last 6 quarters (optional, will build if not provided)
        market_share_table: Market share table (optional)
        prop_holdings_table: Proprietary holdings table (optional)
        investment_composition_table: Investment book composition table (optional)
        return_prompt: If True, returns tuple (commentary, prompt) instead of just commentary

    Returns:
        Generated commentary string, or tuple (commentary, prompt) if return_prompt=True
    """

    # Check for cached commentary first (if not forcing regeneration)
    cache_file = "sql/ai_commentary_cache.csv"
    cached_commentary = None
    if not force_regenerate and os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file, quoting=1)
            cached = cache_df[(cache_df['TICKER'] == ticker) &
                             (cache_df['QUARTER'] == year_quarter)]
            if not cached.empty:
                cached_commentary = cached.iloc[-1]['COMMENTARY']
        except Exception as e:
            print(f"Could not read from cache: {e}")
            pass  # Continue to generate new commentary

    # Use provided analysis_table or build it
    if analysis_table is None or analysis_table.empty:
        # Fallback: build from scratch using old method
        df_6q = get_last_6_quarters_data(ticker, year_quarter, df)
        if df_6q.empty:
            return f"No financial data available for {ticker} in {year_quarter}"
        display_df = df_6q
        margin_equity_ratio = 0
    else:
        # Use the pre-built table from the page (which has proper formatting/annualization)
        display_df = analysis_table.copy()

        # Calculate margin/equity ratio from the latest quarter column
        # Table structure: Metric | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | QoQ% | YoY%
        quarter_cols = [col for col in display_df.columns if col not in ['Metric', 'QoQ Growth %', 'YoY Growth %']]
        if len(quarter_cols) > 0:
            latest_quarter_col = quarter_cols[-1]  # Last quarter column (most recent)

            # Get Margin Balance and calculate ratio
            margin_row = display_df[display_df['Metric'] == 'Margin Balance']
            if not margin_row.empty:
                margin_balance = margin_row[latest_quarter_col].values[0]
                # For ratio calculation, we'd need equity - but this is optional
                margin_equity_ratio = 0  # Will be shown from data if available
            else:
                margin_equity_ratio = 0
        else:
            margin_equity_ratio = 0

    # Create AI prompt with improved brokerage-focused structure
    prompt = f"""
You are a financial analyst specializing in the brokerage sector. Analyze quarterly broker results from the provided data (financial metrics and ratios).

IMPORTANT NOTES ABOUT THE DATA:
- ROE values shown are ALREADY ANNUALIZED (multiplied by 4 for quarterly data)
- All figures are in VND billions (B VND) unless shown as percentages
- Margin/Equity % shows margin lending balance as % of shareholder equity (legal cap is 200%)
- The table shows the last 6 quarters with growth rates

Data for Broker: {ticker} (Quarter: {year_quarter})
{display_df.to_markdown(index=True, tablefmt='grid')}
"""

    # Add market share table if available
    if market_share_table is not None and not market_share_table.empty:
        prompt += f"""
Market Share Data:
{market_share_table.to_markdown(index=False, tablefmt='grid')}
"""

    # Add prop holdings table if available
    if prop_holdings_table is not None and not prop_holdings_table.empty:
        prompt += f"""
Top Proprietary Holdings:
{prop_holdings_table.to_markdown(index=False, tablefmt='grid')}
"""

    # Add investment composition table if available
    if investment_composition_table is not None and not investment_composition_table.empty:
        prompt += f"""
Investment Book Composition (Selected Quarter: {year_quarter}):
{investment_composition_table.to_markdown(index=False, tablefmt='grid')}
"""

    # Add TOI drivers analysis if available
    if toi_drivers_qoq is not None and not toi_drivers_qoq.empty:
        growth_pct_qoq = toi_drivers_qoq.attrs.get('growth_pct', 0)
        prior_q_qoq = toi_drivers_qoq.attrs.get('prior_quarter', '')
        prompt += f"""
TOI Drivers Analysis (Quarter-over-Quarter):
TOI Growth: {growth_pct_qoq:+.1f}% vs {prior_q_qoq}

{toi_drivers_qoq.to_markdown(index=False, tablefmt='grid')}

NOTE: The "Impact (pp)" column shows each income stream's contribution to TOI growth in percentage points.
For example, if Net Brokerage Income shows +5.2pp, it means brokerage contributed 5.2 percentage points to the total TOI growth.
All impacts sum to the total TOI growth rate. The "% of TOI" column shows what percentage each income stream represents of Total Operating Income.
"""

    if toi_drivers_yoy is not None and not toi_drivers_yoy.empty:
        growth_pct_yoy = toi_drivers_yoy.attrs.get('growth_pct', 0)
        prior_q_yoy = toi_drivers_yoy.attrs.get('prior_quarter', '')
        prompt += f"""
TOI Drivers Analysis (Year-over-Year):
TOI Growth: {growth_pct_yoy:+.1f}% vs {prior_q_yoy}

{toi_drivers_yoy.to_markdown(index=False, tablefmt='grid')}

NOTE: The "Impact (pp)" column shows each income stream's contribution to TOI growth in percentage points.
The "% of TOI" column shows what percentage each income stream represents of Total Operating Income.
"""
    prompt += """
Your answer must follow this structure exactly. Do not add or remove sections.


## 1. Overall (max 5 bullet points)
Provide a clear narrative of how the quarter unfolded and how sustainable the results appear.  
State the absolute PBT and TOI, and identify which income streams — brokerage, margin, investment, or IB — were the main contributors or detractors to TOI.  
Use the TOI driver table internally to determine which drivers are material, but do not cite the pp values directly.  
Compare both QoQ and YoY trends and interpret the direction correctly, describing the quarter as strong, stable, or soft in line with internal thresholds.  
Conclude with a cohesive view of what defined the quarter and whether current revenue momentum seems sustainable or driven by short-term factors.

## 2. Traditional brokerage (max 3 bullet points)
Explain brokerage income change and what drove it.
Show absolute market share and whether it improved or declined (ignore QoQ/YoY %).
Break down drivers: liquidity (avg daily trading value), market share, and net brokerage fee — specify which mattered most.
If fee improved, show the absolute fee level and scale of change.
For margin lending, discuss income and balance growth, linking income change to lending volume vs. rate.
Include absolute margin balance for scale.
When interpreting the margin/equity ratio, use this exact logic:
• Below 70% → low level, ample room to expand margin lending.
• 70–150% → **normal** level, operating comfortably within the cap.
• 150–200% → high level, limited remaining capacity before hitting the 200% legal cap.
• Above 200% → exceeds the legal cap — clearly state that this breaches the regulatory limit.
Always compare the current level to the previous quarter and comment on available headroom versus the 200% cap.
Do not refer to this ratio as “leverage.”

## 3. Investment (max 3 bullet points)
Discuss investment income growth QoQ and YoY in context of portfolio mix.
Distinguish accounting groups (FVTPL, AFS, HTM) from underlying assets (equities, bonds, funds, CDs/deposits).
Explain which assets dominate and what that implies for income stability.
Bonds/CDs → stable but low-upside; equities → volatile but high-upside.
If large listed equity exposure, name top holdings and their effect.
Compare the current portfolio mix with the previous quarter to highlight any notable shifts — for example, a higher share of bonds or deposits indicates a more conservative stance, while increased equity exposure suggests a move toward higher risk and volatility.  
If listed equity exposure is meaningful, cite the top holdings and how they influenced results.  
Conclude with whether the overall mix signals a stable, more cautious positioning or a pivot toward greater market sensitivity.

## 4. IB (max 2 bullet points)
Cover IB only if QoQ growth >30%.
Summarize QoQ and YoY trends and whether growth came from deal recovery or major mandates.

## 5. Cost control (max 3 bullet points)
Show SG&A growth QoQ and YoY and discuss CIR trend.
Explain interest expense, rate, and borrowing balance growth, and whether they align with margin lending expansion.
Assess margin lending spread = margin rate – funding cost.
If spread stable/rising, higher borrowing is normal; only flag risk if spread narrows or profitability compresses.
Summarize if costs are offset by revenue momentum or hurting earnings.

Writing rules:  
Each bullet should blend data and interpretation in a single, well-developed sentence.  
Lead with insight and follow with numbers; avoid mechanical or list-like phrasing.  
Maintain a factual, neutral, and professional investor tone.
Do not expand or define common financial abbreviations (e.g., CIR, TOI, SG&A, IB, PBT, ROE, ROA).

Logical rules (apply to all sections):  
• Interpret direction correctly — negative = decline, positive = growth.  
• Do not describe a change from –5% to –14% as an improvement; it is a deeper contraction.  
• Discuss only material items (≥0.5pp contribution or clearly meaningful to performance).  

Formatting:  
Use one decimal place for percentages (e.g., 15.7%). Keep layout clean and consistent.
"""

    # If we have cached commentary and not forcing regeneration, return it now
    if cached_commentary is not None:
        if return_prompt:
            return (cached_commentary, prompt)
        return cached_commentary

    try:
        # Generate commentary using OpenAI
        client = get_openai_client()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst specializing in Vietnamese securities and brokerage firms. You MUST follow the exact structure provided in the prompt. Do not deviate from the requested format."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5
        )

        commentary = response.choices[0].message.content

        # Cache the result
        cache_data = {
            'TICKER': ticker,
            'QUARTER': year_quarter,
            'COMMENTARY': commentary,
            'GENERATED_DATE': datetime.now().isoformat(),
            'MODEL': model
        }

        # Save to cache
        try:
            if os.path.exists(cache_file):
                cache_df = pd.read_csv(cache_file, quoting=1)  # QUOTE_ALL for proper handling
                cache_df = pd.concat([cache_df, pd.DataFrame([cache_data])], ignore_index=True)
            else:
                cache_df = pd.DataFrame([cache_data])

            # Use quoting to handle newlines and quotes in commentary
            cache_df.to_csv(cache_file, index=False, quoting=1)
        except Exception as e:
            print(f"Could not save to cache: {e}")

        # Return tuple if prompt requested, else just commentary
        if return_prompt:
            return (commentary, prompt)
        return commentary

    except Exception as e:
        error_msg = f"Error generating commentary: {str(e)}"
        if return_prompt:
            return (error_msg, prompt if 'prompt' in locals() else "Prompt not generated due to error")
        return error_msg

def get_available_tickers(df: pd.DataFrame) -> list:
    """Get list of available broker tickers from the data."""
    # Get unique tickers and filter out NaN/None values
    tickers = df['TICKER'].dropna().unique()

    # Convert to string and filter out non-string values
    valid_tickers = []
    for ticker in tickers:
        ticker_str = str(ticker).strip()
        # Only include valid ticker strings (3-4 letter codes typically)
        if ticker_str and ticker_str.isalpha() and len(ticker_str) >= 2:
            valid_tickers.append(ticker_str)

    return sorted(valid_tickers)

def get_available_quarters(df: pd.DataFrame, ticker: str = None) -> list:
    """Get list of available quarters, optionally filtered by ticker."""
    if ticker:
        df = df[df['TICKER'] == ticker]

    # Get quarters from QUARTER_LABEL column, excluding 'Annual' entries
    quarters = df[df['QUARTER_LABEL'].notna() & (df['QUARTER_LABEL'] != 'Annual')]['QUARTER_LABEL'].unique()

    # Clean and filter quarters - much stricter filtering
    valid_quarters = []
    for q in quarters:
        # Convert to string and strip whitespace
        q_str = str(q).strip()

        # Only accept quarters that match pattern: digit + Q + 2digits (e.g., "1Q24", "2Q23")
        if (len(q_str) >= 3 and
            q_str[0].isdigit() and
            q_str[1] == 'Q' and
            q_str[2:].isdigit() and
            len(q_str[2:]) >= 1):
            valid_quarters.append(q_str)

    # Sort quarters properly (newest first)
    def quarter_sort_key(quarter):
        try:
            # Handle formats like '1Q24', '2Q23', etc.
            parts = quarter.split('Q')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                quarter_num = int(parts[0])
                year_str = parts[1]

                # Handle 2-digit years (assume 2000s if <= 50, otherwise 1900s)
                if len(year_str) == 2:
                    year_int = int(year_str)
                    if year_int <= 50:  # 00-50 = 2000-2050
                        year = 2000 + year_int
                    else:  # 51-99 = 1951-1999
                        year = 1900 + year_int
                else:
                    year = int(year_str)

                return (year, quarter_num)
            return (0, 0)
        except (ValueError, IndexError):
            return (0, 0)

    # Remove duplicates and sort
    valid_quarters = list(set(valid_quarters))
    return sorted(valid_quarters, key=quarter_sort_key, reverse=True)
