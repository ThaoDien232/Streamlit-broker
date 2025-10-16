"""
Simplified Investment Book Classification for Brokerage Firms
Maps database KEYCODEs to 4 simplified asset groups with market values.

Simplified Categories:
- Marked-to-market equities (FVTPL Equities)
- Not marked-to-market equities (AFS + HTM Equities)  
- Bonds (All bond types)
- CDs/Deposits (All deposit types)

Based on DATABASE_SCHEMA_1.md lines 635-677 (7.x and 8.x series KEYCODEs)
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional

# ============================================================================
# SIMPLIFIED INVESTMENT CATEGORY MAPPING
# ============================================================================

# Maps actual database KEYCODE to simplified category
INVESTMENT_CLASSIFICATION = {
    # MTM Equities (FVTPL - 8.x series market value)
    '8.1.1.1. Listed Shares': 'MTM Equities',
    '8.1.1.2. Unlisted Shares': 'MTM Equities',
    '8.1.2. Fund Certificates': 'MTM Equities',
    '8.1.3. Other Equity Securities': 'MTM Equities',
    
    # Non-MTM Equities (AFS + HTM - 7.x series market value)
    '7.1.1.1. Listed Shares': 'Non-MTM Equities',
    '7.1.1.2. Unlisted Shares': 'Non-MTM Equities', 
    '7.1.2. Fund Certificates': 'Non-MTM Equities',
    '7.1.3. Other Equity Securities': 'Non-MTM Equities',
    
    # Bonds (All bond types - 7.x and 8.x series)
    '7.2.1. Government Bonds': 'Bonds',
    '7.2.2. Municipal Bonds': 'Bonds',
    '7.2.3. Corporate Bonds': 'Bonds',
    '7.2.4. Other Debt Securities': 'Bonds',
    '8.2.1. Government Bonds': 'Bonds',
    '8.2.2. Municipal Bonds': 'Bonds', 
    '8.2.3. Corporate Bonds': 'Bonds',
    '8.2.4. Other Debt Securities': 'Bonds',
    
    # CDs/Deposits (All deposit types)
    '7.3. Term Deposits': 'CDs/Deposits',
    '8.3. Term Deposits': 'CDs/Deposits',
}

# Display order for the 4 categories
SIMPLIFIED_CATEGORIES = [
    'MTM Equities',
    'Non-MTM Equities', 
    'Bonds',
    'CDs/Deposits'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def classify_investment(keycode: str) -> Optional[str]:
    """
    Classify an investment based on KEYCODE.

    Args:
        keycode: Database KEYCODE (e.g., '8.1.1.1. Listed Shares')

    Returns:
        Simplified category name or None if not an investment code
    """
    return INVESTMENT_CLASSIFICATION.get(keycode)

def get_investment_data(df: pd.DataFrame, ticker: str, year: int, quarter: int) -> Dict[str, float]:
    """
    Extract simplified investment holdings for a broker in a specific period.

    Args:
        df: DataFrame with brokerage metrics (must have KEYCODE and VALUE columns)
        ticker: Broker ticker
        year: Year
        quarter: Quarter (1-4) or 5 for annual

    Returns:
        Dictionary with market values for the 4 simplified categories
    """
    # Filter data for this broker/period
    period_data = df[
        (df['TICKER'] == ticker) &
        (df['YEARREPORT'] == year) &
        (df['LENGTHREPORT'] == quarter)
    ].copy()

    # Initialize simplified categories
    investment_book = {category: 0.0 for category in SIMPLIFIED_CATEGORIES}

    # Classify each row
    for _, row in period_data.iterrows():
        keycode = row['KEYCODE']
        category = classify_investment(keycode)

        if category:
            value = row.get('VALUE', 0)
            if pd.notna(value) and value != 0:
                investment_book[category] += value

    return investment_book

def format_investment_book(
    current_data: Dict[str, float],
    prior_data: Dict[str, float] = None,
    current_label: str = "Current",
    prior_label: str = "Prior Quarter"
) -> pd.DataFrame:
    """
    Format simplified investment data with optional prior quarter comparison.

    Args:
        current_data: Current period investment data from get_investment_data()
        prior_data: Prior quarter investment data (optional)
        current_label: Label for current period columns
        prior_label: Label for prior period columns

    Returns:
        Formatted DataFrame with market values in billions VND
    """
    rows = []

    # Build column headers
    if prior_data:
        columns = ['Asset Group', f'{current_label} (B VND)', f'{prior_label} (B VND)', 'Change (B VND)', 'Change (%)']
    else:
        columns = ['Asset Group', 'Market Value (B VND)']

    for category in SIMPLIFIED_CATEGORIES:
        current_value = current_data.get(category, 0)
        current_value_b = current_value / 1_000_000_000

        if prior_data:
            prior_value = prior_data.get(category, 0)
            prior_value_b = prior_value / 1_000_000_000
            change_b = current_value_b - prior_value_b
            change_pct = ((current_value - prior_value) / prior_value * 100) if prior_value != 0 else 0

            row = {
                'Asset Group': category,
                f'{current_label} (B VND)': f'{current_value_b:,.2f}' if current_value != 0 else '-',
                f'{prior_label} (B VND)': f'{prior_value_b:,.2f}' if prior_value != 0 else '-',
                'Change (B VND)': f'{change_b:+,.2f}' if (current_value != 0 or prior_value != 0) else '-',
                'Change (%)': f'{change_pct:+,.1f}%' if (current_value != 0 or prior_value != 0) else '-'
            }
        else:
            row = {
                'Asset Group': category,
                'Market Value (B VND)': f'{current_value_b:,.2f}' if current_value != 0 else '-'
            }

        # Only include rows with data
        if current_value != 0 or (prior_data and prior_data.get(category, 0) != 0):
            rows.append(row)

    # Add total row
    if rows:
        current_total = sum(current_data.values()) / 1_000_000_000
        
        if prior_data:
            prior_total = sum(prior_data.values()) / 1_000_000_000
            total_change_b = current_total - prior_total
            total_change_pct = ((sum(current_data.values()) - sum(prior_data.values())) / sum(prior_data.values()) * 100) if sum(prior_data.values()) != 0 else 0

            total_row = {
                'Asset Group': 'TOTAL INVESTMENTS',
                f'{current_label} (B VND)': f'{current_total:,.2f}',
                f'{prior_label} (B VND)': f'{prior_total:,.2f}',
                'Change (B VND)': f'{total_change_b:+,.2f}',
                'Change (%)': f'{total_change_pct:+,.1f}%'
            }
        else:
            total_row = {
                'Asset Group': 'TOTAL INVESTMENTS',
                'Market Value (B VND)': f'{current_total:,.2f}'
            }
        
        rows.append(total_row)

    return pd.DataFrame(rows)

def get_category_total(investment_data: Dict[str, float], category: str) -> float:
    """
    Get total for a specific simplified category.

    Args:
        investment_data: Dictionary from get_investment_data()
        category: One of the SIMPLIFIED_CATEGORIES

    Returns:
        Total value in VND
    """
    return investment_data.get(category, 0.0)

def get_total_investments(investment_data: Dict[str, float]) -> float:
    """
    Get total investment value across all simplified categories.

    Args:
        investment_data: Dictionary from get_investment_data()

    Returns:
        Total value in VND
    """
    return sum(investment_data.values())
