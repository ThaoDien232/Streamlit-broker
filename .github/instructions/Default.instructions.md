# Coding Guidelines for Vietnamese Brokerage Financial Analysis Project

## 1. Be Calculation-Focused

Prioritize mathematical correctness and clarity in all financial calculations:

- **Use vectorized pandas operations** for performance and clarity
- **Don't add excessive try/except blocks** - assume data exists and is in expected format
- **Focus on the math** - let pandas raise natural errors if data is missing
- **Trust the data structure** - assume Combined_Financial_Data.csv has the expected columns

Example:
```python
# Good - direct, clear calculation
df['ROE'] = df['NPAT'] / df['Total_Equity']

# Avoid - over-defensive
try:
    if 'NPAT' in df.columns and 'Total_Equity' in df.columns:
        if not df.empty:
            df['ROE'] = df['NPAT'] / df['Total_Equity']
except Exception as e:
    print(f"Error: {e}")
```

## 2. Be Direct in Data Analysis

Keep data processing straightforward and readable:

```python
# Good - direct calculation
df['metric'] = df['revenue'] / df['assets']

# Avoid - over-engineered
def calculate_metric(df):
    if 'revenue' not in df.columns:
        raise ValueError("Missing revenue column")
    if 'assets' not in df.columns:
        raise ValueError("Missing assets column")
    if df.empty:
        raise ValueError("DataFrame is empty")
    # ... more checks
    return df['revenue'] / df['assets']
```

**Principles:**
- Write clear, direct pandas operations
- Avoid unnecessary function wrappers for simple calculations
- Let pandas errors surface naturally
- Only add validation where business logic requires it

## 3. Be Succinct in Naming

Use clear but concise naming conventions:

**Financial Metrics:**
- Use standard abbreviations: `ROE`, `ROA`, `NPAT`, `PBT`, `EBITDA`
- Use full names for clarity when needed: `Net_Brokerage_Income`, `Margin_Balance`

**DataFrame Variables:**
- Keep short and contextual: `df_q` (quarterly), `df_a` (annual), `df_calc` (calculated metrics)
- Use `df` for the main working dataframe

**Columns:**
- Match CSV column names exactly: `TICKER`, `YEARREPORT`, `LENGTHREPORT`, `METRIC_CODE`
- Use snake_case for derived columns: `qoq_growth`, `yoy_growth`

Example:
```python
# Good
df_q = df[df['LENGTHREPORT'].isin([1, 2, 3, 4])]
roe = df_calc['NPAT'] / df_calc['Total_Equity']

# Avoid
quarterly_dataframe = financial_data[financial_data['LENGTHREPORT'].isin([1, 2, 3, 4])]
return_on_equity = calculated_metrics_dataframe['NPAT'] / calculated_metrics_dataframe['Total_Equity']
```

## 4. Professional Tone - No Emojis

Keep this project professional and clean:

- **Do not use emojis** in code, comments, variable names, or UI text
- Use clear, professional language in all documentation
- Exception: User-facing UI elements that were already designed with emojis may keep them

```python
# Good
st.subheader("Financial Analysis")
st.button("Generate Analysis")

# Avoid
st.subheader("=Ê Financial Analysis")
st.button("=€ Generate Analysis")
```

## 5. Financial Data Conventions

**Annualization:**
- Quarterly ROE/ROA must be annualized by multiplying by 4
- Annual data (LENGTHREPORT = 5) should not be modified
- Check quarter type before applying annualization

**Number Formatting:**
- Display financial amounts in billions VND: "1,234.5B VND"
- Display percentages with 2 decimal places: "12.38%"
- Use thousand separators for readability
- ROE and ROA are stored as decimals (0.1238) but displayed as percentages (12.38%)

**Growth Calculations:**
- QoQ (Quarter-over-Quarter): Compare to immediately previous quarter (1Q24 vs 4Q23)
- YoY (Year-over-Year): Compare to same quarter previous year (1Q24 vs 1Q23)
- Do not calculate growth for ROE and ROA

## 6. Code Organization

- Keep calculation logic close to where it's used
- Extract common patterns into small, focused functions
- Prefer composition over deep inheritance
- Write self-documenting code with clear variable names

## Summary

**DO:**
- Write clear, direct pandas operations
- Use standard financial abbreviations
- Trust the data structure
- Keep code professional and clean
- Focus on mathematical correctness

**DON'T:**
- Add excessive error handling
- Over-engineer simple calculations
- Use long, verbose names
- Add emojis to code or UI
- Create unnecessary abstraction layers

This project is a financial analysis tool for Vietnamese brokerage firms. Clarity, correctness, and professionalism are paramount.
