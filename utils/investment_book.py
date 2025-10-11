"""
Investment Book Classification for Brokerage Firms
Maps database KEYCODEs (7.x, 8.x series) to investment categories.

Categories:
- FVTPL (Fair Value Through Profit or Loss) - Trading assets
- AFS (Available for Sale) - Long-term holdings at fair value through OCI
- HTM (Held to Maturity) - Investments held to maturity at amortized cost

Based on DATABASE_SCHEMA.md lines 413-453 (7.x and 8.x series KEYCODEs)
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional

# ============================================================================
# INVESTMENT CATEGORY MAPPING (from DATABASE_SCHEMA.md)
# ============================================================================

# Maps KEYCODE to (Category, Instrument Type, Display Name, Valuation Basis)
INVESTMENT_CLASSIFICATION = {
    # FVTPL - Trading Securities (7.1.x and 8.1.x)
    '7.1.1. Trading Securities': ('FVTPL', 'Trading Securities', 'Trading Securities - Total', 'Cost'),
    '7.1.1.1. Listed Shares': ('FVTPL', 'Equities', 'Listed Shares', 'Cost'),
    '7.1.1.2. Unlisted Shares': ('FVTPL', 'Equities', 'Unlisted Shares', 'Cost'),
    '7.1.1.3. Fund': ('FVTPL', 'Equities', 'Fund Certificates', 'Cost'),
    '7.1.1.4. Bonds': ('FVTPL', 'Bonds', 'Bonds', 'Cost'),
    '7.1.2. Other short-term investments': ('FVTPL', 'Others', 'CDs & Short-term Deposits', 'Cost'),

    '8.1.1. Trading Securities': ('FVTPL', 'Trading Securities', 'Trading Securities - Total', 'Market Value'),
    '8.1.1.1. Listed Shares': ('FVTPL', 'Equities', 'Listed Shares', 'Market Value'),
    '8.1.1.2. Unlisted Shares': ('FVTPL', 'Equities', 'Unlisted Shares', 'Market Value'),
    '8.1.1.3. Fund': ('FVTPL', 'Equities', 'Fund Certificates', 'Market Value'),
    '8.1.1.4. Bonds': ('FVTPL', 'Bonds', 'Bonds', 'Market Value'),
    '8.1.2. Other short-term investments': ('FVTPL', 'Others', 'CDs & Short-term Deposits', 'Market Value'),

    # AFS - Available for Sale (7.2.x and 8.2.x)
    '7.2.1. Listed shares': ('AFS', 'Equities', 'Listed Shares', 'Cost'),
    '7.2.2. Unlisted shares': ('AFS', 'Equities', 'Unlisted Shares', 'Cost'),
    '7.2.3. Investment Fund Certificates': ('AFS', 'Equities', 'Fund Certificates', 'Cost'),
    '7.2.4. Bonds': ('AFS', 'Bonds', 'Bonds', 'Cost'),
    '7.2.5. Monetary market instrument': ('AFS', 'Money Market', 'Money Market Instruments', 'Cost'),

    '8.2.1. Listed shares': ('AFS', 'Equities', 'Listed Shares', 'Market Value'),
    '8.2.2. Unlisted shares': ('AFS', 'Equities', 'Unlisted Shares', 'Market Value'),
    '8.2.3. Investment Fund Certificates': ('AFS', 'Equities', 'Fund Certificates', 'Market Value'),
    '8.2.5. Bonds': ('AFS', 'Bonds', 'Bonds', 'Market Value'),

    # HTM - Held to Maturity (7.3.x and 8.5.x)
    '7.3.1. Listed shares': ('HTM', 'Equities', 'Listed Shares', 'Cost'),
    '7.3.2. Unlisted shares': ('HTM', 'Equities', 'Unlisted Shares', 'Cost'),
    '7.3.3. Investment Fund Certificates': ('HTM', 'Equities', 'Fund Certificates', 'Cost'),
    '7.3.4. Bonds': ('HTM', 'Bonds', 'Bonds', 'Cost'),
    '7.3.5. Monetary market instrument': ('HTM', 'Money Market', 'Deposits & Money Market', 'Cost'),

    '8.5.1.1. Listed Shares': ('HTM', 'Equities', 'Listed Shares', 'Market Value'),
    '8.5.1.2. Unlisted Shares': ('HTM', 'Equities', 'Unlisted Shares', 'Market Value'),
    '8.5.1.3. Fund': ('HTM', 'Equities', 'Fund Certificates', 'Market Value'),
    '8.5.1.4. Bonds': ('HTM', 'Bonds', 'Bonds', 'Market Value'),
}

# ============================================================================
# INVESTMENT BOOK STRUCTURE
# ============================================================================

INVESTMENT_BOOK_STRUCTURE = {
    'FVTPL': {
        'display_name': 'Fair Value Through Profit or Loss (Trading Assets)',
        'sections': [
            ('Equities', ['Listed Shares', 'Unlisted Shares', 'Fund Certificates']),
            ('Bonds', ['Bonds']),
            ('Others', ['CDs & Short-term Deposits'])
        ]
    },
    'AFS': {
        'display_name': 'Available-for-Sale Financial Assets',
        'sections': [
            ('Equities', ['Listed Shares', 'Unlisted Shares', 'Fund Certificates']),
            ('Bonds', ['Bonds']),
            ('Money Market', ['Money Market Instruments'])
        ]
    },
    'HTM': {
        'display_name': 'Held-to-Maturity Investments',
        'sections': [
            ('Equities', ['Listed Shares', 'Unlisted Shares', 'Fund Certificates']),
            ('Bonds', ['Bonds']),
            ('Money Market', ['Deposits & Money Market'])
        ]
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_investment(keycode: str) -> Optional[Tuple[str, str, str, str]]:
    """
    Classify an investment based on KEYCODE.

    Args:
        keycode: Database KEYCODE (e.g., '7.1.1.1. Listed Shares')

    Returns:
        Tuple of (Category, Instrument Type, Display Name, Valuation Basis)
        or None if not an investment code
    """
    return INVESTMENT_CLASSIFICATION.get(keycode)

def get_investment_data(df: pd.DataFrame, ticker: str, year: int, quarter: int) -> Dict:
    """
    Extract investment holdings for a broker in a specific period.

    Args:
        df: DataFrame with brokerage metrics (must have KEYCODE and VALUE columns)
        ticker: Broker ticker
        year: Year
        quarter: Quarter (1-4) or 5 for annual

    Returns:
        Dictionary with investment data organized by category
    """
    # Filter data for this broker/period
    period_data = df[
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter)
    ].copy()

    # Build investment book
    investment_book = {
        'FVTPL': {},
        'AFS': {},
        'HTM': {}
    }

    # Classify each row
    for _, row in period_data.iterrows():
        keycode = row['KEYCODE']
        classification = classify_investment(keycode)

        if classification:
            category, instrument_type, display_name, valuation = classification
            value = row.get('VALUE', 0)

            if pd.notna(value) and value != 0:
                # Store by display name and valuation
                if category not in investment_book:
                    investment_book[category] = {}

                if valuation not in investment_book[category]:
                    investment_book[category][valuation] = {}

                investment_book[category][valuation][display_name] = value

    return investment_book

def format_investment_book(
    current_data: Dict,
    prior_data: Dict = None,
    current_label: str = "Current",
    prior_label: str = "Prior Quarter"
) -> pd.DataFrame:
    """
    Format investment data with optional prior quarter comparison.

    - HTM category shows NO G/L column (measured at amortized cost)
    - FVTPL and AFS show G/L column (measured at fair value)

    Args:
        current_data: Current period investment data from get_investment_data()
        prior_data: Prior quarter investment data (optional)
        current_label: Label for current period columns
        prior_label: Label for prior period columns

    Returns:
        Formatted DataFrame with Cost, Market Value, and (for FVTPL/AFS) G/L columns
    """
    rows = []

    for category in ['FVTPL', 'AFS', 'HTM']:
        if category not in current_data or category not in INVESTMENT_BOOK_STRUCTURE:
            continue

        curr_cat = current_data[category]
        prior_cat = prior_data.get(category, {}) if prior_data else {}
        structure = INVESTMENT_BOOK_STRUCTURE[category]

        # Build column headers
        if prior_data:
            header = {
                'Item': category,
                f'{current_label} - Cost': '',
                f'{current_label} - MV': '',
            }
            if category != 'HTM':  # No G/L for HTM
                header[f'{current_label} - G/L'] = ''
            header.update({
                f'{prior_label} - Cost': '',
                f'{prior_label} - MV': '',
            })
            if category != 'HTM':
                header[f'{prior_label} - G/L'] = ''
        else:
            header = {
                'Item': category,
                'Cost (B VND)': '',
                'Market Value (B VND)': '',
            }
            if category != 'HTM':
                header['Unrealized G/L (B VND)'] = ''

        rows.append(header)

        # Process items
        for section_name, items in structure['sections']:
            for item in items:
                curr_cost = curr_cat.get('Cost', {}).get(item, 0)
                curr_mv = curr_cat.get('Market Value', {}).get(item, 0)

                # Skip if no data
                if curr_cost == 0 and curr_mv == 0:
                    if not prior_data:
                        continue
                    prior_cost = prior_cat.get('Cost', {}).get(item, 0)
                    prior_mv = prior_cat.get('Market Value', {}).get(item, 0)
                    if prior_cost == 0 and prior_mv == 0:
                        continue

                curr_cost_b = curr_cost / 1_000_000_000
                curr_mv_b = curr_mv / 1_000_000_000
                curr_gl_b = curr_mv_b - curr_cost_b

                if prior_data:
                    prior_cost = prior_cat.get('Cost', {}).get(item, 0)
                    prior_mv = prior_cat.get('Market Value', {}).get(item, 0)
                    prior_cost_b = prior_cost / 1_000_000_000
                    prior_mv_b = prior_mv / 1_000_000_000
                    prior_gl_b = prior_mv_b - prior_cost_b

                    row = {
                        'Item': f'  {item}',
                        f'{current_label} - Cost': f'{curr_cost_b:,.2f}' if curr_cost != 0 else '-',
                        f'{current_label} - MV': f'{curr_mv_b:,.2f}' if curr_mv != 0 else '-',
                    }
                    if category != 'HTM':
                        row[f'{current_label} - G/L'] = f'{curr_gl_b:+,.2f}' if (curr_cost != 0 or curr_mv != 0) else '-'

                    row.update({
                        f'{prior_label} - Cost': f'{prior_cost_b:,.2f}' if prior_cost != 0 else '-',
                        f'{prior_label} - MV': f'{prior_mv_b:,.2f}' if prior_mv != 0 else '-',
                    })
                    if category != 'HTM':
                        row[f'{prior_label} - G/L'] = f'{prior_gl_b:+,.2f}' if (prior_cost != 0 or prior_mv != 0) else '-'
                else:
                    row = {
                        'Item': f'  {item}',
                        'Cost (B VND)': f'{curr_cost_b:,.2f}',
                        'Market Value (B VND)': f'{curr_mv_b:,.2f}',
                    }
                    if category != 'HTM':
                        row['Unrealized G/L (B VND)'] = f'{curr_gl_b:+,.2f}'

                rows.append(row)

        # Category total
        curr_cat_cost = sum(curr_cat.get('Cost', {}).values())
        curr_cat_mv = sum(curr_cat.get('Market Value', {}).values())

        if curr_cat_cost != 0 or curr_cat_mv != 0:
            curr_cat_cost_b = curr_cat_cost / 1_000_000_000
            curr_cat_mv_b = curr_cat_mv / 1_000_000_000
            curr_cat_gl_b = curr_cat_mv_b - curr_cat_cost_b

            if prior_data:
                prior_cat_cost = sum(prior_cat.get('Cost', {}).values())
                prior_cat_mv = sum(prior_cat.get('Market Value', {}).values())
                prior_cat_cost_b = prior_cat_cost / 1_000_000_000
                prior_cat_mv_b = prior_cat_mv / 1_000_000_000
                prior_cat_gl_b = prior_cat_mv_b - prior_cat_cost_b

                total_row = {
                    'Item': f'Total {category}',
                    f'{current_label} - Cost': f'{curr_cat_cost_b:,.2f}',
                    f'{current_label} - MV': f'{curr_cat_mv_b:,.2f}',
                }
                if category != 'HTM':
                    total_row[f'{current_label} - G/L'] = f'{curr_cat_gl_b:+,.2f}'

                total_row.update({
                    f'{prior_label} - Cost': f'{prior_cat_cost_b:,.2f}',
                    f'{prior_label} - MV': f'{prior_cat_mv_b:,.2f}',
                })
                if category != 'HTM':
                    total_row[f'{prior_label} - G/L'] = f'{prior_cat_gl_b:+,.2f}'
            else:
                total_row = {
                    'Item': f'Total {category}',
                    'Cost (B VND)': f'{curr_cat_cost_b:,.2f}',
                    'Market Value (B VND)': f'{curr_cat_mv_b:,.2f}',
                }
                if category != 'HTM':
                    total_row['Unrealized G/L (B VND)'] = f'{curr_cat_gl_b:+,.2f}'

            rows.append(total_row)

        # Blank row
        rows.append({k: '' for k in header.keys()})

    return pd.DataFrame(rows)

def get_category_total(investment_data: Dict, category: str, valuation: str = 'Market Value') -> float:
    """
    Get total for a specific category and valuation basis.

    Args:
        investment_data: Dictionary from get_investment_data()
        category: 'FVTPL', 'AFS', or 'HTM'
        valuation: 'Cost' or 'Market Value'

    Returns:
        Total value in VND
    """
    if category not in investment_data:
        return 0.0

    return sum(investment_data[category].get(valuation, {}).values())

def get_total_investments(investment_data: Dict) -> float:
    """
    Get total investment value across all categories (at market value).

    Args:
        investment_data: Dictionary from get_investment_data()

    Returns:
        Total value in VND
    """
    total = 0.0
    for category in ['FVTPL', 'AFS', 'HTM']:
        # Use Market Value for FVTPL/AFS, Cost for HTM
        if category == 'HTM':
            total += get_category_total(investment_data, category, 'Cost')
        else:
            total += get_category_total(investment_data, category, 'Market Value')
    return total
