# Profit/Loss Calculation Explained

## Overview
The dashboard calculates profit/loss by comparing the current market value of holdings to their value at the quarter-end.

## Step-by-Step Calculation

### Step 1: Determine Volume (Shares Held)

```
Volume = FVTPL value / Quarter_End_Price
```

**Example:**
- FVTPL value: 10,000,000 VND
- Quarter-End Price (Sept 30): 64.17 VND/share
- Volume = 10,000,000 / 64.17 = 155,847 shares

**Key Point:** The system **assumes you still hold the same number of shares** from the quarter-end to today.

---

### Step 2: Calculate Quarter-End Market Value

```
Quarter_End_Market_Value = Volume × Quarter_End_Price
```

**Example:**
- Volume: 155,847 shares
- Quarter-End Price: 64.17 VND
- Quarter_End_Market_Value = 155,847 × 64.17 = 10,000,000 VND

**Note:** This should equal the FVTPL value (it's a verification step).

---

### Step 3: Calculate Current Market Value

```
Current_Market_Value = Volume × Current_Price
```

**Example:**
- Volume: 155,847 shares (same as quarter-end)
- Current Price (today): 63.20 VND
- Current_Market_Value = 155,847 × 63.20 = 9,849,530 VND

---

### Step 4: Calculate Profit/Loss (Absolute)

```
Profit_Loss = Current_Market_Value - FVTPL value
```

**Example:**
- Current_Market_Value: 9,849,530 VND
- FVTPL value: 10,000,000 VND
- Profit_Loss = 9,849,530 - 10,000,000 = **-150,470 VND (Loss)**

---

### Step 5: Calculate Profit/Loss Percentage

```
Profit_Loss_Pct = (Profit_Loss / Quarter_End_Market_Value) × 100
```

**Example:**
- Profit_Loss: -150,470 VND
- Quarter_End_Market_Value: 10,000,000 VND
- Profit_Loss_Pct = (-150,470 / 10,000,000) × 100 = **-1.5%**

---

## Alternative Formula (Simplified)

Since Quarter_End_Market_Value ≈ FVTPL value, you can also calculate:

```
Profit_Loss_Pct = ((Current_Price - Quarter_End_Price) / Quarter_End_Price) × 100
```

**Example:**
- Current Price: 63.20 VND
- Quarter-End Price: 64.17 VND
- Profit_Loss_Pct = ((63.20 - 64.17) / 64.17) × 100 = **-1.51%**

---

## Code Location

The calculation is performed in two places:

### 1. Main calculation function (lines 178-197)
File: `pages/1_Prop_Book_Dashboard.py`

```python
def calculate_profit_loss(df, quarter_prices, current_prices, quarter):
    # Calculate volume from FVTPL value and quarter-end price
    df_calc['Volume'] = df_calc['FVTPL value'] / df_calc['Quarter_End_Price']

    # Calculate market values
    df_calc['Quarter_End_Market_Value'] = df_calc['Volume'] * df_calc['Quarter_End_Price']
    df_calc['Current_Market_Value'] = df_calc['Volume'] * df_calc['Current_Price']

    # Calculate profit/loss
    df_calc['Profit_Loss'] = df_calc['Current_Market_Value'] - df_calc['FVTPL value']

    # Calculate percentage
    df_calc['Profit_Loss_Pct'] = (df_calc['Profit_Loss'] / df_calc['Quarter_End_Market_Value']) * 100
```

### 2. Pivot table calculation (lines 99-127)
Simplified version for the pivot table display.

---

## Important Assumptions

1. **Constant Volume**: Assumes you hold the same number of shares from quarter-end to today
2. **No Trading Costs**: Does not account for transaction fees or taxes
3. **Mark-to-Market**: Uses closing prices (not bid/ask spread)
4. **Quarter-End Dates**: Uses March 31, June 30, Sept 30, Dec 31
5. **Price Dates**: Uses last trading day on or before quarter-end if market is closed

---

## Why Might the % Be "Slightly Off"?

### Possible Reasons:

1. **Different Quarter-End Date in Excel**
   - Your Excel might use a different date (e.g., last trading day of month vs. exact date)
   - Check what date the FVTPL value in your Excel is based on

2. **Price Source Difference**
   - VCI data might differ slightly from your original TCBS source
   - End-of-day price timing differences

3. **Volume Changes**
   - If you actually bought/sold shares between quarter-end and today
   - The calculation assumes constant volume

4. **Rounding Differences**
   - FVTPL value in Excel might be rounded
   - Price data might have different decimal precision

5. **Corporate Actions**
   - Stock splits, dividends, bonus shares not accounted for

---

## How to Verify

To check if the calculation is correct:

1. Find the ticker in your Excel (e.g., VNM in Q3 2024)
2. Note the FVTPL value (e.g., 10,000,000 VND)
3. Get the quarter-end price (Sept 30, 2024)
4. Get today's price
5. Calculate manually: `((Current_Price - Quarter_End_Price) / Quarter_End_Price) * 100`
6. Compare with dashboard percentage

If there's a discrepancy, the issue is likely in the **price data source** or **quarter-end date**.
