# Database Migration Summary

## ✅ Completed Tasks

### 1. **KEYCODE Mapping Created** ([utils/keycode_mapping.py](utils/keycode_mapping.py))
- Built complete mapping between CSV format (UPPERCASE_UNDERSCORE) and database format (Mixed_Case)
- Verified Balance Sheet totals:
  - `TOTAL_ASSETS` → `BS.92` (TOTAL ASSETS)
  - `TOTAL_EQUITY` → `BS.142` (OWNER'S EQUITY)
- Mapped all 169 Balance Sheet items: `BSA1-BSA169` → `BS.1-BS.169`
- Defined calculated metrics that need computation: ROE, ROA, INTEREST_RATE

### 2. **Database Query Modules Created**

#### [utils/brokerage_data.py](utils/brokerage_data.py)
- `load_brokerage_metrics()` - Load brokerage financial data from database (replaces CSV)
- `get_calc_metric_value()` - Get specific metric values with automatic KEYCODE translation
- `calculate_metric()` - Calculate ROE, ROA, INTEREST_RATE from database values
- `get_available_tickers()` - Get broker list from database
- `get_available_quarters()` - Get available quarters from database
- All queries filter for 2017 onwards only ✅

#### [utils/market_index_data.py](utils/market_index_data.py)
- `load_market_index()` - Load VNINDEX data from MarketIndex table (replaces INDEX.csv)
- `load_market_liquidity_data()` - Calculate quarterly average daily turnover
- `get_quarterly_turnover()` - Get specific quarter turnover
- `get_market_turnover_stats()` - Get yearly statistics

### 3. **Pages Updated to Use Database**

| Page | Status | Changes Made |
|------|--------|--------------|
| [pages/3_Historical.py](pages/3_Historical.py) | ✅ Updated | Uses `load_brokerage_metrics()` instead of CSV |
| [pages/2_Charts.py](pages/2_Charts.py) | ✅ Updated | Uses `load_brokerage_metrics()` instead of CSV |
| [pages/7_AI_Commentary.py](pages/7_AI_Commentary.py) | ✅ Updated | Uses database for both brokerage and market data |
| [pages/6_Valuation.py](pages/6_Valuation.py) | ✅ Already using DB | No changes needed |
| [pages/4_Forecast.py](pages/4_Forecast.py) | ⏸️ Not updated | May still use FORECAST.csv (verify if needed) |

### 4. **Key Features Preserved**
- ✅ Caching with `@st.cache_data(ttl=3600)` for performance
- ✅ Data filtered from 2017 onwards only
- ✅ Quarterly vs Annual data separation (`LENGTHREPORT 1-4` vs `5`)
- ✅ Quarter label parsing (`1Q24`, `2Q25` format)
- ✅ Calculated metrics: ROE, ROA, INTEREST_RATE with proper annualization
- ✅ All existing DataFrame column names preserved for compatibility

---

## 📝 Critical Mappings Reference

### Income Statement Metrics
```python
'NET_BROKERAGE_INCOME'    → 'Net_Brokerage_Income'
'MARGIN_LENDING_INCOME'   → 'Net_Margin_lending_Income'
'NET_INVESTMENT_INCOME'   → 'Net_investment_income'
'NET_TRADING_INCOME'      → 'Net_Trading_Income'
'NET_IB_INCOME'           → 'Net_IB_Income'
'FX_GAIN_LOSS'            → 'FX_Income'
'SGA'                     → 'SG_A'
'PBT'                     → 'PBT' (same)
'NPAT'                    → 'NPAT' (same)
```

### Balance Sheet Metrics
```python
'TOTAL_ASSETS'            → 'BS.92'  (TOTAL ASSETS)
'TOTAL_EQUITY'            → 'BS.142' (OWNER'S EQUITY)
'BORROWING_BALANCE'       → 'Borrowing_Balance'
'MARGIN_BALANCE'          → 'Margin_Lending_book'
```

### Calculated Metrics (Not in Database)
```python
'ROE'                     → Calculate: NPAT / TOTAL_EQUITY * 100 (annualized)
'ROA'                     → Calculate: NPAT / TOTAL_ASSETS * 100 (annualized)
'INTEREST_RATE'           → Calculate: INTEREST_EXPENSE / AVG_BORROWING * 4
```

---

## 🧪 Testing

### Test Script Created
[scripts/test_database_migration.py](scripts/test_database_migration.py)

**Run this to verify migration:**
```bash
python scripts/test_database_migration.py
```

**What it tests:**
1. ✅ Database connection and data loading
2. ✅ Metric value comparison (database vs CSV)
3. ✅ Market liquidity data from MarketIndex table
4. ✅ Available tickers from database
5. ✅ Available quarters from database

### Expected Test Results
- All SSI 2024 Q1 values should match between database and CSV
- Market liquidity should load from MarketIndex table
- ~24-30 broker tickers should be available
- Quarters should be sorted newest first (2Q25, 1Q25, 4Q24, etc.)

---

## 🔄 Next Steps

### Immediate (Before Going Live)
1. **Run test script** to verify database values match CSV
   ```bash
   python scripts/test_database_migration.py
   ```

2. **Test each page individually** in Streamlit:
   - Historical page: Check if data loads and displays correctly
   - Charts page: Verify charts render with database data
   - AI Commentary: Ensure commentary generation works

3. **Verify edge cases:**
   - ROE/ROA calculations are correct (annualized for quarterly)
   - Interest rate calculation uses average borrowing
   - Market liquidity shows correct billions VND values

### After Verification
4. **Archive CSV files** (don't delete immediately):
   ```bash
   mkdir archive
   mv sql/Combined_Financial_Data.csv archive/
   mv sql/INDEX.csv archive/
   ```

5. **Remove obsolete calculation scripts** (if all calculations work from DB):
   - `utils/calculate_new_metrics.py` (may no longer be needed)

6. **Update documentation:**
   - Update CLAUDE.md to reflect database usage
   - Add database connection requirements to README

---

## 🔧 Database Connection

### Required Configuration
Database credentials are in [.streamlit/secrets.toml](.streamlit/secrets.toml):
```toml
[db]
server = "dcdwhprod.database.windows.net"
database = "dclab"
username = "dclab_readonly"
password = "DHS#@vGESADdf!"
```

### Tables Used
1. `dbo.BrokerageMetrics` - Financial statements and calculated metrics
2. `dbo.MarketIndex` - VNINDEX daily data and turnover
3. `dbo.Market_Data` - Valuation ratios (already in use)

---

## ⚠️ Known Limitations

1. **KEYCODE Name Differences:**
   - Some metrics have different names in database vs CSV
   - Mapping layer handles translation automatically

2. **Calculated Metrics:**
   - ROE, ROA, INTEREST_RATE are NOT in database
   - Calculated on-the-fly using component values
   - Properly annualized for quarterly data (multiply by 4)

3. **Data Availability:**
   - Database queries are slower than CSV (but cached for 1 hour)
   - First load may take 10-30 seconds depending on query complexity
   - Subsequent loads are instant (cached)

4. **Quarter Label Format:**
   - Database uses same format as CSV: `1Q24`, `2Q25`
   - Parsing logic preserved from original code

---

## 📊 Performance Considerations

### Caching Strategy
- **Brokerage data:** Cached for 1 hour (`ttl=3600`)
- **Market liquidity:** Cached for 30 minutes (`ttl=1800`)
- **Valuation data:** Cached for 1 hour (existing)

### Query Optimization
- All queries filter for `YEARREPORT >= 2017` to reduce data volume
- Quarterly data excludes annual (`LENGTHREPORT BETWEEN 1 AND 4`)
- Ticker-specific queries when possible for faster results

---

## 🎯 Migration Checklist

- [x] Build KEYCODE mapping dictionary
- [x] Create database query modules (brokerage_data.py, market_index_data.py)
- [x] Update Historical page to use database
- [x] Update Charts page to use database
- [x] Update AI Commentary page to use database
- [x] Find exact KEYCODEs for TOTAL_ASSETS and TOTAL_EQUITY
- [x] Create test script for validation
- [ ] Run test script and verify data accuracy
- [ ] Test all pages in Streamlit
- [ ] Archive CSV files after verification
- [ ] Update project documentation

---

## 📞 Support

If database queries fail or timeout:
1. Check database connection in [.streamlit/secrets.toml](.streamlit/secrets.toml)
2. Verify network connectivity to Azure SQL
3. Check query timeout settings in [utils/db.py](utils/db.py)
4. Review error messages in Streamlit console

For KEYCODE mapping issues:
- See [utils/keycode_mapping.py](utils/keycode_mapping.py) for all mappings
- Add new mappings if additional metrics are discovered
- Update calculation formulas in `CALCULATED_METRICS` dict
