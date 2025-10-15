"""
KEYCODE Mapping between our CSV format and Dragon Capital database format.
Based on DATABASE_SCHEMA_1.md documentation.
"""

# ============================================================================
# CALCULATED METRICS MAPPING
# Maps our METRIC_CODE (uppercase_underscore) to database KEYCODE (Mixed_Case)
# ============================================================================

METRIC_TO_DB_KEYCODE = {
    # Income Statement - Revenue Streams (DATABASE_SCHEMA_1.md lines 684-694)
    'Net_Brokerage_Income': 'Net_Brokerage_Income',  # line 684
    'NET_IB_INCOME': 'Net_IB_Income',  # line 685
    'NET_TRADING_INCOME': 'Net_Trading_Income',  # line 686 - Fixed!
    'NET_INVESTMENT_INCOME': 'Net_investment_income',  # line 693
    'MARGIN_LENDING_INCOME': 'Net_Margin_lending_Income',  # line 688
    'NET_OTHER_OP_INCOME': 'Net_other_operating_income',  # line 689 - Fixed!
    'NET_OTHER_INCOME': 'Net_Other_Income',  # line 690
    'FEE_INCOME': 'Net_Fee_Income',  # line 691
    'CAPITAL_INCOME': 'Net_Capital_Income',  # line 692
    'TOTAL_OPERATING_INCOME': 'Total_Operating_Income',  # line 694

    # Income Statement - Other Items
    'FX_GAIN_LOSS': 'FX_Income',
    'AFFILIATES_DIVESTMENT': 'Affiliates_Divestment',
    'ASSOCIATES_INCOME': 'Associate_Income',  # Note: singular in DB
    'DEPOSIT_INCOME': 'Deposit_Income',
    'INTEREST_INCOME': 'Net_Interest_Income',
    'INTEREST_EXPENSE': 'Interest_Expense',

    # Income Statement - Expenses
    'SGA': 'SG_A',  # Sales, General & Administrative

    # Profitability
    'PBT': 'PBT',  # Profit Before Tax (same)
    'NPAT': 'NPAT',  # Net Profit After Tax (same)

    # Balance Sheet - Balances
    'BORROWING_BALANCE': 'Borrowing_Balance',
    'MARGIN_BALANCE': 'Margin_Lending_book',  # Fixed: correct DB keycode is Margin_Lending_book (BS.8)

    # Balance Sheet - Totals (verified from CSV: BSA53→BS.92, BSA78→BS.142)
    'TOTAL_ASSETS': 'BS.92',  # TOTAL ASSETS
    'TOTAL_EQUITY': 'BS.142',  # OWNER'S EQUITY

    # Trading Values (Note items - NOS101-110 section per DATABASE_SCHEMA_1.md)
    # KEYCODE column shows actual database values (descriptive names, not NOS codes)
    'INSTITUTION_SHARES_TRADING_VALUE': 'Institution_shares_trading_value',
    'INSTITUTION_BOND_TRADING_VALUE': 'Institution_bond_trading_value',
    'INVESTOR_SHARES_TRADING_VALUE': 'Investor_shares_trading_value',
    'INVESTOR_BOND_TRADING_VALUE': 'Investor_bond_trading_value',

    # Ratio Metrics - Now available directly in database (DATABASE_SCHEMA_1.md lines 707-714)
    'ROE': 'ROE',  # Return on Equity (annualized)
    'ROA': 'ROA',  # Return on Assets (annualized)
    'INTEREST_RATE': 'INTEREST_RATE',  # Borrowing Interest Rate (annualized)
    'NET_BROKERAGE_FEE': 'NET_BROKERAGE_FEE',  # Net Brokerage Fee (bps)
    'MARGIN_LENDING_RATE': 'MARGIN_LENDING_RATE',  # Margin Lending Rate (annualized)
    'MARGIN_EQUITY_RATIO': 'MARGIN_EQUITY_RATIO',  # Margin to Equity Ratio
}

# ============================================================================
# BALANCE SHEET MAPPING (Raw KEYCODEs)
# Maps our BSA format to database BS. format
# ============================================================================

# Generate BS mappings programmatically
# Our CSV uses: BSA1, BSA2, BSA3, ..., BSA169
# Database uses: BS.1, BS.2, BS.3, ..., BS.169
for i in range(1, 170):
    METRIC_TO_DB_KEYCODE[f'BSA{i}'] = f'BS.{i}'

# ============================================================================
# REVERSE MAPPING (for compatibility)
# ============================================================================

DB_KEYCODE_TO_METRIC = {
    v: k for k, v in METRIC_TO_DB_KEYCODE.items()
    if v is not None  # Exclude None values (calculated metrics)
}

# ============================================================================
# INVESTMENT PORTFOLIO KEYCODES (7.x and 8.x series)
# These are hierarchical and include spaces/dots
# ============================================================================

PORTFOLIO_KEYCODES = {
    # Cost of Financial Investment (7.x series)
    '7. Cost of the financial investment': 'Total investment cost',
    '7.1. Short-term investments': 'FVTPL short-term',
    '7.1.1. Trading Securities': 'Trading portfolio cost',
    '7.1.1.1. Listed Shares': 'FVTPL listed equity cost',
    '7.1.1.2. Unlisted Shares': 'FVTPL unlisted equity cost',
    '7.1.1.3. Fund': 'FVTPL fund cost',
    '7.1.1.4. Bonds': 'FVTPL bonds cost',
    '7.1.2. Other short-term investments': 'Other FVTPL cost',
    '7.2.1. Listed shares': 'AFS listed equity cost',
    '7.2.2. Unlisted shares': 'AFS unlisted equity cost',
    '7.2.3. Investment Fund Certificates': 'AFS funds cost',
    '7.2.4. Bonds': 'AFS bonds cost',
    '7.2.5. Monetary market instrument': 'AFS money market cost',
    '7.3.1. Listed shares': 'HTM listed equity cost',
    '7.3.2. Unlisted shares': 'HTM unlisted equity cost',
    '7.3.3. Investment Fund Certificates': 'HTM funds cost',
    '7.3.4. Bonds': 'HTM bonds cost',
    '7.3.5. Monetary market instrument': 'HTM money market cost',

    # Market Value of Financial Investment (8.x series)
    '8. Market value of financial investment': 'Total investment market value',
    '8.1. Long-term financial investment': 'FVTPL market value',
    '8.1.1. Trading Securities': 'Trading portfolio FV',
    '8.1.1.1. Listed Shares': 'FVTPL listed equity FV',
    '8.1.1.2. Unlisted Shares': 'FVTPL unlisted equity FV',
    '8.1.1.3. Fund': 'FVTPL fund FV',
    '8.1.1.4. Bonds': 'FVTPL bonds FV',
    '8.1.2. Other short-term investments': 'Other FVTPL FV',
    '8.2.1. Listed shares': 'AFS listed equity FV',
    '8.2.2. Unlisted shares': 'AFS unlisted equity FV',
    '8.2.3. Investment Fund Certificates': 'AFS funds FV',
    '8.2.5. Bonds': 'AFS bonds FV',
    '8.5.1.1. Listed Shares': 'Long-term listed FV',
    '8.5.1.2. Unlisted Shares': 'Long-term unlisted FV',
    '8.5.1.3. Fund': 'Long-term fund FV',
    '8.5.1.4. Bonds': 'Long-term bonds FV',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_keycode(metric_code: str) -> str:
    """
    Translate our METRIC_CODE to database KEYCODE.

    Args:
        metric_code: Our format (e.g., 'NET_BROKERAGE_INCOME')

    Returns:
        Database KEYCODE (e.g., 'Net_Brokerage_Income')
    """
    return METRIC_TO_DB_KEYCODE.get(metric_code)

def get_metric_code(db_keycode: str) -> str:
    """
    Translate database KEYCODE to our METRIC_CODE.

    Args:
        db_keycode: Database format (e.g., 'Net_Brokerage_Income')

    Returns:
        Our METRIC_CODE (e.g., 'NET_BROKERAGE_INCOME')
    """
    return DB_KEYCODE_TO_METRIC.get(db_keycode)
