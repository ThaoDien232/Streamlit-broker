"""
Brokerage Earnings Driver Analysis
Calculates contribution of income streams and costs to PBT growth
Similar to Banking_Drivers methodology
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Tuple, Optional


def calculate_earnings_drivers(
    ticker: str,
    current_quarter: str,
    comparison_type: str = 'QoQ'
) -> pd.DataFrame:
    """
    Calculate earnings drivers showing contribution of each component to PBT growth.

    Args:
        ticker: Broker ticker (e.g., 'SSI', 'VCI')
        current_quarter: Quarter label (e.g., '1Q24')
        comparison_type: 'QoQ' (quarter-over-quarter) or 'YoY' (year-over-year)

    Returns:
        DataFrame with earnings driver analysis
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
    """Extract key financial metrics for a specific quarter."""
    quarter_df = df[df['QUARTER_LABEL'] == quarter]

    # Helper function to get value safely
    def get_value(keycode: str) -> float:
        row = quarter_df[quarter_df['KEYCODE'] == keycode]
        if not row.empty:
            return float(row.iloc[0]['VALUE'])
        return 0.0

    return {
        'quarter': quarter,
        'pbt': get_value('PBT'),
        'net_brokerage': get_value('NET_BROKERAGE_INCOME'),
        'margin_lending': get_value('NET_MARGIN_LENDING_INCOME'),
        'investment_income': get_value('NET_INVESTMENT_INCOME'),
        'ib_income': get_value('NET_IB_INCOME'),
        'other_operating': get_value('NET_OTHER_OPERATING_INCOME'),
        'sga': get_value('SG_A'),  # Already negative
        'interest_expense': get_value('INTEREST_EXPENSE'),  # Already negative
        'other_income': get_value('NET_OTHER_INCOME'),
        'total_operating_income': get_value('TOTAL_OPERATING_INCOME')
    }


def _calculate_driver_contributions(
    current: Dict[str, float],
    prior: Dict[str, float],
    comparison_type: str
) -> pd.DataFrame:
    """
    Calculate contribution of each driver to PBT growth.

    Methodology:
    1. Calculate absolute changes for each component
    2. Calculate PBT change and growth %
    3. Normalize each component's contribution to scores (sum to ±100%)
    4. Convert scores to actual impact (percentage points)
    """

    # Step 1: Calculate changes
    changes = {
        'PBT': current['pbt'] - prior['pbt'],
        'Net Brokerage Income': current['net_brokerage'] - prior['net_brokerage'],
        'Margin Lending Income': current['margin_lending'] - prior['margin_lending'],
        'Investment Income': current['investment_income'] - prior['investment_income'],
        'IB Income': current['ib_income'] - prior['ib_income'],
        'Other Operating Income': current['other_operating'] - prior['other_operating'],
        'SG&A': current['sga'] - prior['sga'],  # Change in expense (negative = good)
        'Interest Expense': current['interest_expense'] - prior['interest_expense'],  # Change in expense
        'Other Income': current['other_income'] - prior['other_income']
    }

    # Step 2: Calculate PBT growth %
    pbt_change = changes['PBT']

    if prior['pbt'] == 0:
        growth_pct = 0
    else:
        growth_pct = (pbt_change / abs(prior['pbt'])) * 100

    # Handle small PBT changes to avoid extreme ratios
    pbt_change_abs = abs(pbt_change)
    if pbt_change_abs < 50_000_000:  # Less than 50M VND
        pbt_change_abs = 50_000_000
        small_pbt_flag = True
    else:
        small_pbt_flag = False

    # Step 3: Normalize to scores (each component's contribution)
    # Score = (Component Change / |PBT Change|) × 100
    scores = {}
    for component, change in changes.items():
        if component != 'PBT':
            scores[component] = (change / pbt_change_abs) * 100

    # Step 4: Convert scores to impacts (percentage points)
    # Impact = (Score / 100) × Growth_%
    impacts = {}
    for component, score in scores.items():
        impacts[component] = (score / 100) * abs(growth_pct)
        # Preserve sign based on whether PBT grew or declined
        if pbt_change < 0:
            impacts[component] = -impacts[component]

    # Build output DataFrame
    results = []

    # Revenue components (positive contributions)
    revenue_items = [
        'Net Brokerage Income',
        'Margin Lending Income',
        'Investment Income',
        'IB Income',
        'Other Operating Income'
    ]

    # Cost components (negative = good for PBT)
    cost_items = [
        'SG&A',
        'Interest Expense'
    ]

    # Other
    other_items = ['Other Income']

    # Revenue section
    results.append({
        'Component': '=== REVENUE DRIVERS ===',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': ''
    })

    for item in revenue_items:
        if item == 'Net Brokerage Income':
            curr_val = current['net_brokerage']
            prior_val = prior['net_brokerage']
        elif item == 'Margin Lending Income':
            curr_val = current['margin_lending']
            prior_val = prior['margin_lending']
        elif item == 'Investment Income':
            curr_val = current['investment_income']
            prior_val = prior['investment_income']
        elif item == 'IB Income':
            curr_val = current['ib_income']
            prior_val = prior['ib_income']
        elif item == 'Other Operating Income':
            curr_val = current['other_operating']
            prior_val = prior['other_operating']
        else:
            curr_val = 0
            prior_val = 0

        results.append({
            'Component': item,
            'Current': curr_val / 1e9,  # Convert to billions
            'Prior': prior_val / 1e9,
            'Change': changes[item] / 1e9,
            'Impact (pp)': impacts.get(item, 0)
        })

    # Cost section
    results.append({
        'Component': '=== COST DRIVERS ===',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': ''
    })

    for item in cost_items:
        if item == 'SG&A':
            curr_val = current['sga']
            prior_val = prior['sga']
        elif item == 'Interest Expense':
            curr_val = current['interest_expense']
            prior_val = prior['interest_expense']
        else:
            curr_val = 0
            prior_val = 0

        results.append({
            'Component': item,
            'Current': curr_val / 1e9,
            'Prior': prior_val / 1e9,
            'Change': changes[item] / 1e9,
            'Impact (pp)': impacts.get(item, 0)
        })

    # Other income
    results.append({
        'Component': '=== OTHER ===',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': ''
    })

    for item in other_items:
        curr_val = current['other_income']
        prior_val = prior['other_income']

        results.append({
            'Component': item,
            'Current': curr_val / 1e9,
            'Prior': prior_val / 1e9,
            'Change': changes[item] / 1e9,
            'Impact (pp)': impacts.get(item, 0)
        })

    # Summary
    results.append({
        'Component': '=== SUMMARY ===',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': ''
    })

    results.append({
        'Component': 'PBT',
        'Current': current['pbt'] / 1e9,
        'Prior': prior['pbt'] / 1e9,
        'Change': pbt_change / 1e9,
        'Impact (pp)': growth_pct
    })

    # Calculate total revenue and cost impacts
    total_revenue_impact = sum(impacts.get(item, 0) for item in revenue_items)
    total_cost_impact = sum(impacts.get(item, 0) for item in cost_items)
    total_other_impact = sum(impacts.get(item, 0) for item in other_items)

    results.append({
        'Component': 'Total Revenue Impact',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': total_revenue_impact
    })

    results.append({
        'Component': 'Total Cost Impact',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': total_cost_impact
    })

    results.append({
        'Component': 'Total Other Impact',
        'Current': '',
        'Prior': '',
        'Change': '',
        'Impact (pp)': total_other_impact
    })

    # Create DataFrame
    df_results = pd.DataFrame(results)

    # Add metadata
    df_results.attrs['comparison_type'] = comparison_type
    df_results.attrs['current_quarter'] = current['quarter']
    df_results.attrs['prior_quarter'] = prior['quarter']
    df_results.attrs['growth_pct'] = growth_pct
    df_results.attrs['small_pbt_flag'] = small_pbt_flag

    return df_results


def format_earnings_drivers_table(df: pd.DataFrame) -> str:
    """
    Format earnings drivers DataFrame for display.

    Args:
        df: Earnings drivers DataFrame from calculate_earnings_drivers()

    Returns:
        Formatted markdown table string
    """
    if df.empty:
        return "No earnings driver data available."

    # Get metadata
    comparison = df.attrs.get('comparison_type', 'QoQ')
    current_q = df.attrs.get('current_quarter', '')
    prior_q = df.attrs.get('prior_quarter', '')
    growth = df.attrs.get('growth_pct', 0)

    # Format the table
    formatted = f"### Earnings Drivers Analysis ({comparison})\n"
    formatted += f"**{current_q}** vs **{prior_q}** | PBT Growth: **{growth:.1f}%**\n\n"

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
            table_rows.append(f"**{component}**: {row['Impact (pp)']:.1f}pp")
        else:
            curr = row['Current']
            prior = row['Prior']
            change = row['Change']
            impact = row['Impact (pp)']

            table_rows.append(
                f"- {component}: {curr:.1f}B → {prior:.1f}B "
                f"(Δ{change:+.1f}B) | **{impact:+.1f}pp**"
            )

    formatted += '\n'.join(table_rows)

    return formatted
