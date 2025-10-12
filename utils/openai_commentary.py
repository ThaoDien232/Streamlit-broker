"""
OpenAI integration for generating quarterly business performance commentary.
Uses Combined_Financial_Data.csv and broker-specific analysis.
"""

import os
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

def get_openai_client():
    """Initialize OpenAI client with API key from Streamlit secrets."""
    try:
        api_key = st.secrets["openai"]["api_key"]
    except KeyError:
        # Fallback to environment variable for backward compatibility
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please add it to .streamlit/secrets.toml under [openai] api_key or as OPENAI_API_KEY environment variable.")

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
            cache_df = pd.read_csv(cache_file, quoting=1, escapechar='\\')
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

    prompt += """
Your answer must follow this structure exactly. Do not add or remove sections.

## 1. Overall (max 5 bullet points)
Focus on what kind of quarter this was, what drove it, and how sustainable it looks.
Start with PBT QoQ and YoY growth, then show the ROE beside it to frame profitability.
Describe which specific income stream (brokerage, margin, investment, or IB) primarily drove the quarter’s change.
Do not mention total operating income as a driver since it is a sum of all income streams.
Avoid unrelated income commentary in this section. Use this part to give the investor a clear sense of the overall tone of the quarter (improving, stable, or softening).

## 2. Traditional brokerage (max 3 bullet points)
Present brokerage income growth QoQ and YoY, followed by the trend in market liquidity (average daily trading value).
Conclude whether brokerage income changes were primarily driven by market liquidity or market share.
Then discuss margin lending income and balance growth QoQ and YoY, connecting margin income growth to margin balance growth to show whether it was driven by lending volume or rate changes.
When mentioning margin/equity ratio, include both the current and previous quarter’s level to give context. 
Classify the margin/equity ratio as follows: above 150 percent = high, 70–150 percent = normal, below 70 percent = low. 
Discuss how the current level implies lending capacity or headroom versus the 200 percent legal cap.

## 3. Investment (max 3 bullet points)
Present investment income growth QoQ and YoY and connect it to the composition and any change in the investment book. 
Go beyond FVTPL, AFS, or HTM classification — describe whether the broker mainly holds bonds, equities, or deposits/CDs, as this determines the nature of its investment style. 
A broker with large holdings in bonds, deposits, or CDs typically shows stable investment income, while one with heavy listed equity exposure tends to show higher volatility. 
If the broker has high exposure to listed FVTPL equities, include the top holdings and explain how they may have affected performance. 
Comment on whether recent shifts in the investment mix suggest a more conservative or aggressive investment posture.

## 4. IB (max 2 bullet points)
Only discuss IB income if QoQ growth is greater than 30 percent.
Show IB income QoQ and YoY growth and briefly explain if this came from deal volume recovery or large mandates.

## 5. Cost control (max 3 bullet points)
Present SGA growth QoQ and YoY and discuss CIR trend.
Discuss interest expense growth QoQ and YoY, interest rate changes, and borrowing balance growth.
Highlight if interest rate increased more than 20 bps QoQ or if cost pressure is significant.

Writing Approach Rules:
Each paragraph must combine data and interpretation in one sentence.
Story first, numbers second.
Avoid mechanical bridges and focus on the most material movements.
Tone should be factual, concise, and investment-oriented.

Formatting:
Use one decimal place for percentages (e.g., 15.7%).
Keep output clean and easy to read.
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

            # Use quoting and escapechar to handle newlines and quotes in commentary
            cache_df.to_csv(cache_file, index=False, quoting=1, escapechar='\\')
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