# Data Usage Guide

**Purpose**: Ensure consistent financial metric calculation and display across all pages in the Streamlit Broker application.

**Last Updated**: 2025-10-13

---

## Table of Contents
1. [Core Principles](#core-principles)
2. [Database Schema Quick Reference](#database-schema-quick-reference)
3. [Metric Categories](#metric-categories)
4. [Standard Data Loading Patterns](#standard-data-loading-patterns)
5. [Ratio Cache System](#ratio-cache-system)
6. [Growth Rate Calculations](#growth-rate-calculations)
7. [Display Formatting Standards](#display-formatting-standards)
8. [Common Pitfalls](#common-pitfalls)
9. [Page-Specific Guidelines](#page-specific-guidelines)

---

## Core Principles

### 1. Single Source of Truth
- **All data comes from `BrokerageMetrics` database table**
- Never hardcode calculations that already exist in the database
- Use `utils/brokerage_data.py` functions for all data access
- Reference `DATABASE_SCHEMA_1.md` for authoritative schema documentation

### 2. Consistent Metric Naming
- Database column: `KEYCODE` (e.g., `'ROE'`, `'Net_Brokerage_Income'`, `'BS.92'`)
- Application column: `METRIC_CODE` (same as KEYCODE after loading)
- Use `utils/keycode_mapping.py` for metric definitions
- **Never create custom metric names**

### 3. Separation of Concerns
- **Data Loading**: `utils/brokerage_data.py`
- **Metric Mapping**: `utils/keycode_mapping.py`
- **Earnings Analysis**: `utils/earnings_drivers.py`
- **AI Commentary**: `utils/openai_commentary.py`
- **Display Logic**: Individual page files (`pages/*.py`)

---

## Database Schema Quick Reference

### BrokerageMetrics Table Structure

```
Columns:
- TICKER: Broker code ('SSI', 'VCI', 'HCM', 'Sector')
- YEARREPORT: Year (2024)
- LENGTHREPORT: Quarter (1-4) or 5 for annual
- QUARTER_LABEL: '1Q24', '2Q24', etc.
- KEYCODE: Metric identifier
- KEYCODE_NAME: Human-readable name
- VALUE: Numeric value in VND
- STARTDATE, ENDDATE: Period dates
```

### KEYCODE Categories (244 total)

1. **Balance Sheet (BS.*)**: 169 items
   - `BS.92`: TOTAL ASSETS
   - `BS.142`: OWNER'S EQUITY
   - `BS.8`: Margin Lending Balance
   - `BS.95, BS.100, BS.122, BS.127`: Borrowing components

2. **Income Statement (IS.*)**: Limited items
   - `IS.30`: Provision for losses from mortgage assets

3. **Investment Portfolio (7.x and 8.x)**: 47 items
   - `7.1.1.4. Bonds`: FVTPL bond cost
   - `8.1.1.1. Listed Shares`: FVTPL listed equity market value

4. **Calculated Metrics**: 23+ items (see below)

### Statement Type Classification

After loading, data has `STATEMENT_TYPE` column:
- `'BS'`: Balance Sheet (KEYCODE like 'BS.%')
- `'IS'`: Income Statement (KEYCODE like 'IS.%')
- `'NOTE'`: Investment Portfolio (KEYCODE like '7.%' or '8.%')
- `'CALC'`: Calculated metrics (everything else)

---

## Metric Categories

### Category 1: Pre-calculated Income Statement Metrics

These are **already calculated and stored** in the database with `STATEMENT_TYPE='CALC'`. **Always read directly**, never recalculate.

```python
# Database KEYCODE → Application METRIC_CODE (same name)
INCOME_METRICS = {
    'Net_Brokerage_Income': 'Net_Brokerage_Income',  # IS.10 + IS.33
    'Net_Margin_lending_Income': 'Net_Margin_lending_Income',  # IS.7 + IS.30
    'Net_investment_income': 'Net_investment_income',  # Trading + Interest
    'Net_IB_Income': 'Net_IB_Income',  # IB income streams
    'Net_Other_Income': 'Net_Other_Income',  # IS.52+54+63
    'Total_Operating_Income': 'Total_Operating_Income',  # Capital + Fee
    'Interest_Expense': 'Interest_Expense',  # IS.51
    'SG_A': 'SG_A',  # IS.57 + IS.58
    'PBT': 'PBT',  # IS.65
    'NPAT': 'NPAT',  # IS.71
    'Borrowing_Balance': 'Borrowing_Balance',  # BS.95+100+122+127
    'Margin_Lending_book': 'Margin_Lending_book',  # BS.8
}
```

**Formula Reference** (from DATABASE_SCHEMA_1.md lines 455-480):
- Net_Brokerage_Income = IS.10 + IS.33
- Net_Margin_lending_Income = IS.7 + IS.30
- Net_investment_income = Trading + Interest
- Total_Operating_Income = Capital Income + Fee Income
- Interest_Expense = IS.51
- SG_A = IS.57 + IS.58
- PBT = IS.65
- NPAT = IS.71
- Borrowing_Balance = BS.95 + BS.100 + BS.122 + BS.127

### Category 2: ROE and ROA (Special Case)

**ROE and ROA are stored in the database** with `KEYCODE='ROE'` and `KEYCODE='ROA'`.

```python
# These are pre-calculated and annualized in the database
RATIO_METRICS = {
    'ROE': 'ROE',  # Already in database as KEYCODE='ROE'
    'ROA': 'ROA',  # Already in database as KEYCODE='ROA'
}
```

**Important Notes**:
- ROE/ROA are **already annualized** (multiplied by 4 for quarterly data)
- Values are stored as percentages (e.g., 15.5 means 15.5%)
- Do **not** recalculate from NPAT/TOTAL_EQUITY
- Query with: `KEYCODE='ROE'` and `STATEMENT_TYPE='CALC'`

**Correct Usage**:
```python
from utils.brokerage_data import get_calc_metric_value

# ✅ CORRECT - Read from database
roe = get_calc_metric_value(df, 'SSI', 2024, 1, 'ROE')
roa = get_calc_metric_value(df, 'SSI', 2024, 1, 'ROA')
```

**Incorrect Usage**:
```python
# ❌ WRONG - Don't recalculate
npat = get_calc_metric_value(df, 'SSI', 2024, 1, 'NPAT')
equity = get_calc_metric_value(df, 'SSI', 2024, 1, 'TOTAL_EQUITY')
roe = (npat / equity) * 100 * 4  # DON'T DO THIS
```

### Category 3: Metrics Requiring Calculation

These are **not stored** and must be calculated on-the-fly:

```python
CALCULATED_METRICS = {
    'INTEREST_RATE': {
        'formula': '(Interest_Expense / Borrowing_Balance) * 4',  # Annualized
        'components': ['Interest_Expense', 'Borrowing_Balance'],
    },
    'MARGIN_EQUITY_RATIO': {
        'formula': '(Margin_Lending_book / BS.142) * 100',  # Margin/Equity %
        'components': ['Margin_Lending_book', 'TOTAL_EQUITY'],
    },
    'CIR': {
        'formula': '(SG_A / Total_Operating_Income) * 100',  # Cost-to-Income Ratio
        'components': ['SG_A', 'Total_Operating_Income'],
    },
}
```

**Usage**:
```python
from utils.brokerage_data import calculate_metric

interest_rate = calculate_metric('SSI', 2024, 1, 'INTEREST_RATE', df=df)
```

### Category 4: Market Data (External Sources)

From HSX API, not the database:

```python
EXTERNAL_METRICS = {
    'TRADING_VALUE': 'HSX API - Market trading volume',
    'MARKET_SHARE': 'Calculated: (Brokerage Income * 2) / Trading Value',
    'NET_BROKERAGE_FEE': 'Calculated: (Brokerage Income / Trading Value) * 10000 (bps)',
}
```

---

## Standard Data Loading Patterns

### Pattern 1: AI Commentary Page (Single Ticker, Multiple Quarters)

```python
from utils.brokerage_data import load_ticker_quarter_data

# Load with historical lookback
df = load_ticker_quarter_data(
    ticker='SSI',
    quarter_label='1Q24',
    lookback_quarters=6  # Current + 5 previous quarters
)

# Filter for specific metric
roe_data = df[
    (df['METRIC_CODE'] == 'ROE') &
    (df['STATEMENT_TYPE'] == 'CALC')
]
```

### Pattern 2: Charts/Historical Pages (Multiple Tickers and Metrics)

```python
from utils.brokerage_data import load_filtered_brokerage_data

# Optimized loading - specify exactly what you need
df = load_filtered_brokerage_data(
    tickers=['SSI', 'VCI', 'HCM'],
    metrics=['ROE', 'ROA', 'NET_BROKERAGE_INCOME'],  # Use METRIC_CODE names
    years=[2023, 2024, 2025],
    quarters=[1, 2, 3, 4]
)

# Data is ready to use - METRIC_CODE column is standardized
roe_data = df[df['METRIC_CODE'] == 'ROE']
```

### Pattern 3: Get Single Metric Value

```python
from utils.brokerage_data import get_calc_metric_value

# Best when you need one value
roe = get_calc_metric_value(
    df,  # Optional: pass dataframe to avoid re-querying
    ticker='SSI',
    year=2024,
    quarter=1,
    metric_code='ROE'
)
```

### Pattern 4: Earnings Driver Analysis

```python
from utils.earnings_drivers import calculate_earnings_drivers

# Waterfall analysis of PBT growth drivers
drivers_qoq = calculate_earnings_drivers(
    ticker='SSI',
    current_quarter='1Q24',
    comparison_type='QoQ'  # or 'YoY'
)

# Returns DataFrame with:
# Component | Current | Prior | Change | Impact (pp)
```

---

## Ratio Cache System

### Overview

To improve page load performance, calculated ratios (ROE, ROA, INTEREST_RATE, MARGIN_LENDING_RATE, NET_BROKERAGE_FEE) are pre-calculated and stored in a persistent cache. This avoids expensive runtime calculations on every page load.

**Cache Location**: `cache/calculated_ratios/`
- `ratios_cache.parquet` - Pre-calculated ratio values (Parquet format with Snappy compression)
- `cache_metadata.json` - Version, timestamp, and statistics (fast lookup)

**Current Cache Version**: v2 (INTEREST_RATE now returns percentage, multiplied by 100)

### When to Rebuild Cache

Run the cache build script when:
- ✅ First time setup
- ✅ New quarterly data added to database
- ✅ Calculation logic changes (increment `CACHE_VERSION` in `utils/ratio_cache.py`)
- ✅ User clicks "Refresh Calculation" button in Streamlit UI

### Building the Cache

#### Command Line Usage

```bash
# Build cache (first time or after new data)
python build_ratio_cache.py

# Clear existing cache and rebuild
python build_ratio_cache.py --clear

# Check cache status only (no rebuild)
python build_ratio_cache.py --stats
```

#### Expected Output

```
============================================================
BUILDING RATIO CACHE
============================================================
Started at: 2025-10-15 14:30:00

📋 Step 1: Loading available tickers and years...
   Tickers: 18 (ACBS, BSI, DSE, FTS, HCM...)
   Years: 7 (2019-2025)
   Quarters: [1, 2, 3, 4]

📊 Step 2: Metrics to calculate:
   - ROE
   - ROA
   - INTEREST_RATE
   - MARGIN_LENDING_RATE
   - NET_BROKERAGE_FEE

📥 Step 3: Loading component data from database...
   Components: BORROWING_BALANCE, INTEREST_EXPENSE, MARGIN_BALANCE...
   Loaded: 12,543 component records

🔢 Step 4: Calculating ratios...
   Progress: 10% (5,040/50,400) - 3,245 records created
   Progress: 20% (10,080/50,400) - 6,512 records created
   ...
   ✅ Completed: 50,400 calculations
   ✅ Created: 8,945 ratio records

   Breakdown by metric:
   - ROE: 1,789 records
   - ROA: 1,789 records
   - INTEREST_RATE: 1,789 records
   - MARGIN_LENDING_RATE: 1,789 records
   - NET_BROKERAGE_FEE: 1,789 records

💾 Step 6: Saving to cache...
   ✅ Saved 8,945 records to cache

============================================================
CACHE BUILD COMPLETED
============================================================
Finished at: 2025-10-15 14:32:15

============================================================
RATIO CACHE STATUS
============================================================
Status: VALID
Message: Cache valid - 8,945 records for 18 brokers
Last Updated: 2025-10-15 14:32:15
Cached Metrics: INTEREST_RATE, MARGIN_LENDING_RATE, NET_BROKERAGE_FEE, ROA, ROE
Years: 2019 - 2025
Tickers: 18
============================================================

✅ SUCCESS! Cache is ready to use.
```

### Using the Cache in Code

The cache is automatically used by `load_filtered_brokerage_data()` - no code changes needed:

```python
from utils.brokerage_data import load_filtered_brokerage_data

# This will use cached ratios if available
df = load_filtered_brokerage_data(
    tickers=['SSI', 'VCI'],
    metrics=['ROE', 'ROA', 'INTEREST_RATE'],  # Loaded from cache (fast!)
    years=[2024, 2025],
    quarters=[1, 2, 3, 4]
)
```

**How it works**:
1. Function checks if cache exists and version matches (reads `cache_metadata.json` - very fast)
2. If valid, loads pre-calculated ratios from `ratios_cache.parquet`
3. If invalid/missing, calculates on-the-fly (slower, then caches result)
4. Cache is completely transparent to application code

### Cache Management Functions

```python
from utils.ratio_cache import (
    is_cache_valid,      # Fast check: cache exists and version matches
    load_cached_ratios,  # Load all cached ratios as DataFrame
    save_cached_ratios,  # Save calculated ratios to cache
    clear_cache,         # Delete cache files
    get_cache_info,      # Get human-readable cache status
    print_cache_stats    # Print cache statistics to console
)

# Check if cache is valid (fast - only reads metadata JSON)
if is_cache_valid():
    print("Cache is ready!")
else:
    print("Need to rebuild cache")

# Get cache information for UI display
info = get_cache_info()
print(f"Status: {info['status']}")          # 'valid', 'outdated', 'missing'
print(f"Message: {info['message']}")        # Human-readable description
print(f"Last Updated: {info['last_updated']}")  # Timestamp
```

### Cache File Structure

```
cache/
└── calculated_ratios/
    ├── ratios_cache.parquet      # Main cache data (Parquet with Snappy compression)
    │   Columns:
    │   - TICKER: Broker code (SSI, VCI, HCM, etc.)
    │   - YEARREPORT: Year (2024)
    │   - LENGTHREPORT: Quarter (1-4)
    │   - QUARTER_LABEL: Quarter display label ('1Q24', '2Q24', etc.)
    │   - METRIC_CODE: Ratio name (ROE, ROA, INTEREST_RATE, etc.)
    │   - VALUE: Calculated ratio value
    │
    └── cache_metadata.json        # Fast metadata lookup (small JSON file)
        {
          "version": 2,                    # Cache version (for invalidation)
          "last_updated": "2025-10-15T14:32:15",  # ISO timestamp
          "total_records": 8945,           # Total ratio records
          "tickers": ["ACBS", "BSI", ...], # List of tickers
          "metrics": ["ROE", "ROA", ...],  # List of metrics
          "years": [2019, ..., 2025],      # Years covered
          "num_tickers": 18                # Ticker count
        }
```

### Performance Impact

| Scenario | Without Cache | With Cache | Improvement |
|----------|---------------|------------|-------------|
| **First load** | 15-20s | 15-20s | 0% (builds cache) |
| **Subsequent loads** | 15-20s | 2-3s | **85-90% faster** |
| **After data update** | 15-20s | Click button + 15-20s | Same (one-time rebuild) |
| **Daily usage** | Always slow | Always fast | **Major UX improvement** |

### Troubleshooting

#### Cache shows "outdated" status
- **Cause**: `CACHE_VERSION` in `utils/ratio_cache.py` was incremented (calculation logic changed)
- **Solution**: Run `python build_ratio_cache.py` to rebuild with new logic

#### Cache shows "missing" status
- **Cause**: Cache files deleted or first time setup
- **Solution**: Run `python build_ratio_cache.py`

#### Values look incorrect after code change
- **Cause**: Cache contains old calculations from before code change
- **Solution**:
  1. Increment `CACHE_VERSION` in `utils/ratio_cache.py` (line 18)
  2. Run `python build_ratio_cache.py`

#### Cache build fails with database error
- **Cause**: Database connection issue or missing data
- **Solution**: Check database connection settings and verify data exists in `BrokerageMetrics` table

### Cache Version History

- **v1**: Initial cache implementation
- **v2**: Fixed INTEREST_RATE calculation - now multiplies by 100 to return percentage (consistent with MARGIN_LENDING_RATE)

### Git Configuration

Cache files are excluded from version control:

```
# cache/.gitignore
calculated_ratios/
*.parquet
*.json
*.pkl
```

**Why?** Cache files are large and machine-specific. Each developer/deployment should build their own cache from the database.

---

## Growth Rate Calculations

### Standard Growth Rate Formula

**All pages must use this consistent formula**:

```python
def calculate_growth_rate(current: float, prior: float) -> float:
    """
    Standard growth rate calculation.
    Returns None for invalid inputs.
    """
    if pd.isna(current) or pd.isna(prior) or prior == 0:
        return None
    return ((current - prior) / abs(prior)) * 100
```

### Growth Rate Types

```python
# QoQ (Quarter-over-Quarter): Compare to previous quarter
qoq_growth = calculate_growth_rate(q2_value, q1_value)

# YoY (Year-over-Year): Compare to same quarter last year
yoy_growth = calculate_growth_rate(q1_2024, q1_2023)

# Sequential quarters: 1Q24, 2Q24, 3Q24, 4Q24, 1Q25
# YoY for 1Q25: Compare 1Q25 vs 1Q24 (4 quarters back)
```

### Quarter Sorting

```python
def sort_quarters_chronologically(quarters: List[str]) -> List[str]:
    """Sort quarters from oldest to newest."""
    def quarter_sort_key(q):
        try:
            if 'Q' in q:
                parts = q.split('Q')
                quarter_num = int(parts[0])
                year = int(parts[1])
                if year < 50:
                    year += 2000
                else:
                    year += 1900
                return (year, quarter_num)
        except:
            pass
        return (0, 0)

    return sorted(quarters, key=quarter_sort_key)
```

---

## Display Formatting Standards

### Number Formatting

```python
def format_value(value: float, metric_name: str) -> str:
    """Standard formatting for financial metrics."""
    if pd.isna(value) or value == 'N/A':
        return "N/A"

    # Percentages (already in percentage form, e.g., 15.5 = 15.5%)
    if metric_name in ['ROE', 'ROA', 'CIR', 'Interest Rate',
                       'Margin/Equity %', 'Brokerage Market Share']:
        return f"{value:.2f}%"

    # Basis points
    elif metric_name == 'Net Brokerage Fee':
        return f"{value:.2f} bps"

    # Trading Value (already in billions)
    elif metric_name == 'Trading Value':
        return f"{value:,.1f}B VND"

    # Large financial values (convert to billions)
    elif abs(value) >= 1e9:
        return f"{value/1e9:,.1f}B VND"
    elif abs(value) >= 1e6:
        return f"{value/1e6:,.1f}M VND"
    else:
        return f"{value:,.0f} VND"

def format_growth(value: float) -> str:
    """Format growth rates with sign."""
    if pd.isna(value) or value == 'N/A':
        return "N/A"
    return f"{value:+.1f}%"  # Include sign (+/-)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Recalculating ROE/ROA

**Wrong**:
```python
npat = get_calc_metric_value(df, 'SSI', 2024, 1, 'NPAT')
equity = get_calc_metric_value(df, 'SSI', 2024, 1, 'TOTAL_EQUITY')
roe = (npat / equity) * 100 * 4
```

**Correct**:
```python
roe = get_calc_metric_value(df, 'SSI', 2024, 1, 'ROE')
```

### ❌ Pitfall 2: Inconsistent Metric Names

**Wrong** (different pages using different names):
```python
df[df['METRIC_CODE'] == 'Net Brokerage Income']  # Missing underscore
df[df['KEYCODE'] == 'net_brokerage_income']      # Wrong case
```

**Correct** (use exact KEYCODE from database):
```python
df[df['METRIC_CODE'] == 'Net_Brokerage_Income']  # Exact match
```

### ❌ Pitfall 3: Not Filtering by STATEMENT_TYPE

**Wrong**:
```python
roe = df[df['METRIC_CODE'] == 'ROE']['VALUE'].iloc[0]
```

**Correct**:
```python
roe = df[
    (df['METRIC_CODE'] == 'ROE') &
    (df['STATEMENT_TYPE'] == 'CALC')
]['VALUE'].iloc[0]
```

### ❌ Pitfall 4: Inconsistent Growth Rate Formulas

**Wrong** (different formulas on different pages):
```python
growth1 = (current - prior) / prior * 100
growth2 = ((current / prior) - 1) * 100
growth3 = (current - prior) / abs(current) * 100  # VERY WRONG!
```

**Correct** (use standard function):
```python
growth = calculate_growth_rate(current, prior)
# Formula: ((current - prior) / abs(prior)) * 100
```

### ❌ Pitfall 5: Wrong Mapping in keycode_mapping.py

**Wrong**:
```python
METRIC_TO_DB_KEYCODE = {
    'ROE': None,  # WRONG - ROE exists in database!
    'ROA': None,  # WRONG - ROA exists in database!
}
```

**Correct**:
```python
METRIC_TO_DB_KEYCODE = {
    'ROE': 'ROE',  # CORRECT - maps to KEYCODE='ROE'
    'ROA': 'ROA',  # CORRECT - maps to KEYCODE='ROA'
}
```

---

## Page-Specific Guidelines

### AI Commentary Page (`pages/7_AI_Commentary.py`)

**Purpose**: Generate AI-powered quarterly business analysis

**Data Loading**:
```python
from utils.brokerage_data import load_ticker_quarter_data
df = load_ticker_quarter_data('SSI', '1Q24', lookback_quarters=6)
```

**Key Features**:
- Earnings driver analysis (`utils/earnings_drivers.py`)
- Market share calculation (HSX API + brokerage income)
- Proprietary holdings display
- Investment composition breakdown
- OpenAI commentary generation

**Metrics Used**: All income statement metrics, ROE, ROA, margin metrics, market data

### Charts Page (`pages/2_Charts.py`)

**Purpose**: Visualize time series for multiple tickers

**Data Loading**:
```python
from utils.brokerage_data import load_filtered_brokerage_data
df = load_filtered_brokerage_data(tickers, metrics, years, quarters)
```

**Requirements**:
- ✅ Must support ROE, ROA (read from database)
- ✅ Filter by `METRIC_CODE` column
- ✅ Handle missing data gracefully
- ✅ Consistent color scheme across charts
- ✅ Chronological quarter sorting

### Historical Page (`pages/3_Historical.py`)

**Purpose**: Display tabular historical data

**Data Loading**:
```python
from utils.brokerage_data import load_filtered_brokerage_data
df = load_filtered_brokerage_data(tickers, metrics, years, quarters)
```

**Requirements**:
- ✅ Must support ROE, ROA (read from database)
- ✅ Pivot table: quarters as columns, metrics as rows
- ✅ Apply consistent formatting by metric type
- ✅ Sort quarters chronologically
- ✅ Handle "N/A" values properly

---

## Migration Checklist

When updating a page to follow this guide:

- [ ] Replace custom data loading with `utils/brokerage_data.py` functions
- [ ] Use `METRIC_CODE` consistently for all filtering
- [ ] Remove any manual ROE/ROA calculations
- [ ] Use `get_calc_metric_value()` for single metric values
- [ ] Apply standard `format_value()` and `format_growth()` functions
- [ ] Use standard `calculate_growth_rate()` function
- [ ] Filter by `STATEMENT_TYPE='CALC'` when querying calculated metrics
- [ ] Test with ROE, ROA, and major income statement metrics
- [ ] Verify quarter sorting is chronological (oldest to newest)
- [ ] Ensure number formatting matches standards

---

## Quick Reference

### Essential Imports

```python
# Data loading
from utils.brokerage_data import (
    load_filtered_brokerage_data,  # Multi-ticker, multi-metric
    load_ticker_quarter_data,       # Single ticker, historical
    get_calc_metric_value,          # Single value lookup
    calculate_metric                # Calculate derived metrics
)

# Earnings analysis
from utils.earnings_drivers import calculate_earnings_drivers

# Metric mapping
from utils.keycode_mapping import (
    METRIC_TO_DB_KEYCODE,
    get_db_keycode,
    needs_calculation
)
```

### Key Metrics Quick List

```python
# Pre-calculated in database (STATEMENT_TYPE='CALC')
DATABASE_METRICS = [
    'ROE', 'ROA',  # Already annualized
    'Net_Brokerage_Income',
    'Net_Margin_lending_Income',
    'Net_investment_income',
    'Net_IB_Income',
    'Total_Operating_Income',
    'SG_A',
    'Interest_Expense',
    'PBT',
    'NPAT',
    'Borrowing_Balance',
    'Margin_Lending_book'
]

# Requires calculation
NEEDS_CALCULATION = [
    'INTEREST_RATE',
    'MARGIN_EQUITY_RATIO',
    'CIR'
]

# External sources
EXTERNAL_METRICS = [
    'TRADING_VALUE',
    'MARKET_SHARE',
    'NET_BROKERAGE_FEE'
]
```

---

## Questions?

1. Check `DATABASE_SCHEMA_1.md` for schema documentation
2. Review `utils/keycode_mapping.py` for metric definitions
3. Look at `pages/7_AI_Commentary.py` as reference implementation
4. Update this guide if you discover missing information

---

**Document Owner**: Development Team
**Last Reviewed**: 2025-10-13
