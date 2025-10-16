"""
Brokerage TOI (Total Operating Income) Driver Analysis
Calculates contribution of income streams to TOI growth
Focuses only on revenue components, excluding costs (SG&A, Interest Expense)
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional


def calculate_toi_drivers(
    ticker: str,
    current_quarter: str,
    comparison_type: str = 'QoQ'
) -> pd.DataFrame:
    """
    Calculate TOI drivers showing contribution of each income stream to TOI growth.

    Args:
        ticker: Broker ticker (e.g., 'SSI', 'VCI')
        current_quarter: Quarter label (e.g., '1Q24')
        comparison_type: 'QoQ' (quarter-over-quarter) or 'YoY' (year-over-year)

    Returns:
        DataFrame with TOI driver analysis
    """
    from utils.brokerage_data import load_ticker_quarter_data

    # Determine lookback quarters needed
    lookback = 2 if comparison_type == 'QoQ' else 5  # Need current + 1 prior for QoQ, +4 for YoY

    # Load data
    df = load_ticker_quarter_data(ticker=ticker, quarter_label=current_quarter, lookback_quarters=lookback)

    if df.empty:
        return pd.DataFrame()

    # Get unique quarters sorted
    quarters = sorted(df['QUARTER_LABEL'].unique(),
                     key=lambda q: _quarter_sort_key(q))

    if len(quarters) < 2:
        return pd.DataFrame()  # Need at least 2 quarters to compare

    # Determine current and prior quarter
    current_q = quarters[-1]

    if comparison_type == 'QoQ':
        prior_q = quarters[-2]
    else:  # YoY
        if len(quarters) < 5:
            return pd.DataFrame()  # Need 5 quarters for YoY
        prior_q = quarters[-5]  # 4 quarters back

    # Extract data for both periods
    current_data = _extract_quarter_data(df, current_q)
    prior_data = _extract_quarter_data(df, prior_q)

    # Calculate drivers
    drivers = _calculate_driver_contributions(current_data, prior_data, comparison_type)

    return drivers


def _quarter_sort_key(quarter_label: str) -> Tuple[int, int]:
    """Convert quarter label like '1Q24' to sortable tuple (year, quarter)."""
    try:
        q_num = int(quarter_label[0])
        year = int(quarter_label[2:])
        if year < 50:
            year += 2000
        else:
            year += 1900
        return (year, q_num)
    except:
        return (0, 0)


def _extract_quarter_data(df: pd.DataFrame, quarter: str) -> Dict[str, float]:
    """Extract TOI and income stream metrics for a specific quarter."""
    quarter_df = df[df['QUARTER_LABEL'] == quarter]

    # Helper function to get value safely
    def get_value(metric_code: str) -> float:
        # Try METRIC_CODE first
        row = quarter_df[quarter_df['METRIC_CODE'] == metric_code]
        if not row.empty:
            return float(row.iloc[0]['VALUE'])

        # Fallback to KEYCODE for backward compatibility
        row = quarter_df[quarter_df.get('KEYCODE', pd.Series()) == metric_code]
        if not row.empty:
            return float(row.iloc[0]['VALUE'])

        return 0.0

    return {
        'quarter': quarter,
        'total_operating_income': get_value('Total_Operating_Income'),
        'net_brokerage': get_value('Net_Brokerage_Income'),
        'margin_lending': get_value('Net_Margin_lending_Income'),
        'trading_income': get_value('Net_Trading_Income'),
        'interest_income': get_value('Net_Interest_Income'),
        'ib_income': get_value('Net_IB_Income'),
        'other_operating': get_value('Net_other_operating_income')
    }


def _calculate_driver_contributions(
    current: Dict[str, float],
    prior: Dict[str, float],
    comparison_type: str
) -> pd.DataFrame:
    """
    Calculate contribution of each income stream to TOI growth.

    Methodology:
    1. Calculate absolute changes for each component
    2. Calculate TOI change and growth %
    3. Calculate each component's contribution as % of TOI change
    4. Convert contribution to percentage points impact
    """

    # Step 1: Calculate changes
    changes = {
        'Total Operating Income': current['total_operating_income'] - prior['total_operating_income'],
        'Net Brokerage Income': current['net_brokerage'] - prior['net_brokerage'],
        'Margin Lending Income': current['margin_lending'] - prior['margin_lending'],
        'Trading Income': current['trading_income'] - prior['trading_income'],
        'Interest Income': current['interest_income'] - prior['interest_income'],
        'IB Income': current['ib_income'] - prior['ib_income'],
        'Other Operating Income': current['other_operating'] - prior['other_operating']
    }

    # Step 2: Calculate TOI growth %
    toi_change = changes['Total Operating Income']

    if prior['total_operating_income'] == 0:
        growth_pct = 0
    else:
        growth_pct = (toi_change / abs(prior['total_operating_income'])) * 100

    # Handle small TOI changes to avoid extreme ratios
    toi_change_abs = abs(toi_change)
    if toi_change_abs < 50_000_000:  # Less than 50M VND
        toi_change_abs = 50_000_000
        small_toi_flag = True
    else:
        small_toi_flag = False

    # Step 3: Calculate contribution scores
    # Score = (Component Change / |TOI Change|) × 100
    scores = {}
    for component, change in changes.items():
        if component != 'Total Operating Income':
            scores[component] = (change / toi_change_abs) * 100

    # Step 4: Convert scores to impacts (percentage points)
    # Impact = (Score / 100) × Growth_%
    impacts = {}
    for component, score in scores.items():
        impacts[component] = (score / 100) * abs(growth_pct)
        # Preserve sign based on whether TOI grew or declined
        if toi_change < 0:
            impacts[component] = -impacts[component]

    # Build output DataFrame
    results = []

    # TOI income stream components
    income_items = [
        'Net Brokerage Income',
        'Margin Lending Income',
        'Trading Income',
        'Interest Income',
        'IB Income',
        'Other Operating Income'
    ]

    # Income streams section
    results.append({
        'Component': '=== TOI INCOME STREAMS ===',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': '',
        '% of TOI': ''
    })

    for item in income_items:
        if item == 'Net Brokerage Income':
            curr_val = current['net_brokerage']
            prior_val = prior['net_brokerage']
        elif item == 'Margin Lending Income':
            curr_val = current['margin_lending']
            prior_val = prior['margin_lending']
        elif item == 'Trading Income':
            curr_val = current['trading_income']
            prior_val = prior['trading_income']
        elif item == 'Interest Income':
            curr_val = current['interest_income']
            prior_val = prior['interest_income']
        elif item == 'IB Income':
            curr_val = current['ib_income']
            prior_val = prior['ib_income']
        elif item == 'Other Operating Income':
            curr_val = current['other_operating']
            prior_val = prior['other_operating']
        else:
            curr_val = 0
            prior_val = 0

        # Calculate % of current TOI
        pct_of_toi = (curr_val / current['total_operating_income'] * 100) if current['total_operating_income'] != 0 else 0

        results.append({
            'Component': item,
            'Current': curr_val / 1e9,  # Convert to billions
            'Prior': prior_val / 1e9,
            'Change': changes[item] / 1e9,
            'Impact (pp)': impacts.get(item, 0),
            '% of TOI': pct_of_toi
        })

    # Summary
    results.append({
        'Component': '=== SUMMARY ===',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': '',
        '% of TOI': ''
    })

    results.append({
        'Component': 'Total Operating Income',
        'Current': current['total_operating_income'] / 1e9,
        'Prior': prior['total_operating_income'] / 1e9,
        'Change': toi_change / 1e9,
        'Impact (pp)': growth_pct,
        '% of TOI': 100.0
    })

    # Calculate total income impact (should equal TOI growth)
    total_income_impact = sum(impacts.get(item, 0) for item in income_items)

    results.append({
        'Component': 'Total Income Stream Impact',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': total_income_impact,
        '% of TOI': ''
    })

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Add metadata
    df_results.attrs['comparison_type'] = comparison_type
    df_results.attrs['current_quarter'] = current['quarter']
    df_results.attrs['prior_quarter'] = prior['quarter']
    df_results.attrs['growth_pct'] = growth_pct
    df_results.attrs['small_toi_flag'] = small_toi_flag

    return df_results


def format_toi_drivers_table(df: pd.DataFrame) -> str:
    """
    Format TOI drivers DataFrame for display.

    Args:
        df: TOI drivers DataFrame from calculate_toi_drivers()

    Returns:
        Formatted markdown table string
    """
    if df.empty:
        return "No TOI driver data available."

    # Get metadata
    comparison = df.attrs.get('comparison_type', 'QoQ')
    current_q = df.attrs.get('current_quarter', '')
    prior_q = df.attrs.get('prior_quarter', '')
    growth = df.attrs.get('growth_pct', 0)

    # Format the table
    formatted = f"### TOI Drivers Analysis ({comparison})\n"
    formatted += f"**{current_q}** vs **{prior_q}** | TOI Growth: **{growth:.1f}%**\n\n"

    # Create formatted table
    table_rows = []
    for _, row in df.iterrows():
        component = row['Component']

        # Handle section headers
        if '===' in component:
            table_rows.append(f"\n**{component.replace('=', '').strip()}**\n")
            continue

        # Format values
        if row['Current'] == '':
            # Summary rows
            impact_val = row['Impact (pp)']
            if isinstance(impact_val, (int, float)):
                table_rows.append(f"**{component}**: {impact_val:.1f}pp")
        else:
            curr = row['Current']
            prior = row['Prior']
            change = row['Change']
            impact = row['Impact (pp)']
            pct_toi = row['% of TOI']

            if component == 'Total Operating Income':
                table_rows.append(
                    f"**{component}**: {curr:.1f}B → {prior:.1f}B "
                    f"(Δ{change:+.1f}B) | **{impact:+.1f}%**"
                )
            else:
                table_rows.append(
                    f"- {component}: {curr:.1f}B ({pct_toi:.1f}% of TOI) → {prior:.1f}B "
                    f"(Δ{change:+.1f}B) | **{impact:+.1f}pp**"
                )

    formatted += '\n'.join(table_rows)

    return formatted
