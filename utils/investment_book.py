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
    '7.1.2. Other short-term investments': ('FVTPL', 'Others', 'Other Short-term Investments', 'Cost'),

    '8.1.1. Trading Securities': ('FVTPL', 'Trading Securities', 'Trading Securities - Total', 'Market Value'),
    '8.1.1.1. Listed Shares': ('FVTPL', 'Equities', 'Listed Shares', 'Market Value'),
    '8.1.1.2. Unlisted Shares': ('FVTPL', 'Equities', 'Unlisted Shares', 'Market Value'),
    '8.1.1.3. Fund': ('FVTPL', 'Equities', 'Fund Certificates', 'Market Value'),
    '8.1.1.4. Bonds': ('FVTPL', 'Bonds', 'Bonds', 'Market Value'),
    '8.1.2. Other short-term investments': ('FVTPL', 'Others', 'Other Short-term Investments', 'Market Value'),

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
    '7.3.5. Monetary market instrument': ('HTM', 'Money Market', 'Money Market Instruments', 'Cost'),

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
            ('Others', ['Other Short-term Investments'])
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
            ('Money Market', ['Money Market Instruments'])
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

def format_investment_book(investment_data: Dict, show_category: str = None) -> pd.DataFrame:
    """
    Format investment data into a readable DataFrame.

    Args:
        investment_data: Dictionary from get_investment_data()
        show_category: Specific category to format ('FVTPL', 'AFS', 'HTM') or None for all

    Returns:
        Formatted DataFrame with Cost and Market Value columns
    """
    categories = [show_category] if show_category else ['FVTPL', 'AFS', 'HTM']

    rows = []

    for category in categories:
        if category not in investment_data or category not in INVESTMENT_BOOK_STRUCTURE:
            continue

        cat_data = investment_data[category]
        structure = INVESTMENT_BOOK_STRUCTURE[category]

        # Add category header
        rows.append({
            'Item': f"**{structure['display_name']}**",
            'Cost (B VND)': '',
            'Market Value (B VND)': '',
            'Unrealized G/L (B VND)': ''
        })

        # Process sections
        for section_name, items in structure['sections']:
            section_cost = 0
            section_mv = 0

            for item in items:
                cost = cat_data.get('Cost', {}).get(item, 0)
                mv = cat_data.get('Market Value', {}).get(item, 0)

                if cost != 0 or mv != 0:
                    cost_b = cost / 1_000_000_000
                    mv_b = mv / 1_000_000_000
                    gl_b = mv_b - cost_b

                    rows.append({
                        'Item': f'  {item}',
                        'Cost (B VND)': f'{cost_b:,.2f}',
                        'Market Value (B VND)': f'{mv_b:,.2f}',
                        'Unrealized G/L (B VND)': f'{gl_b:+,.2f}'
                    })

                    section_cost += cost
                    section_mv += mv

            # Section subtotal
            if section_cost != 0 or section_mv != 0:
                cost_b = section_cost / 1_000_000_000
                mv_b = section_mv / 1_000_000_000
                gl_b = mv_b - cost_b

                rows.append({
                    'Item': f'  Subtotal - {section_name}',
                    'Cost (B VND)': f'{cost_b:,.2f}',
                    'Market Value (B VND)': f'{mv_b:,.2f}',
                    'Unrealized G/L (B VND)': f'{gl_b:+,.2f}'
                })

        # Category total
        cat_cost = sum(cat_data.get('Cost', {}).values())
        cat_mv = sum(cat_data.get('Market Value', {}).values())

        if cat_cost != 0 or cat_mv != 0:
            cost_b = cat_cost / 1_000_000_000
            mv_b = cat_mv / 1_000_000_000
            gl_b = mv_b - cost_b

            rows.append({
                'Item': f'**Total {category}**',
                'Cost (B VND)': f'**{cost_b:,.2f}**',
                'Market Value (B VND)': f'**{mv_b:,.2f}**',
                'Unrealized G/L (B VND)': f'**{gl_b:+,.2f}**'
            })

        # Blank row
        rows.append({'Item': '', 'Cost (B VND)': '', 'Market Value (B VND)': '', 'Unrealized G/L (B VND)': ''})

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
