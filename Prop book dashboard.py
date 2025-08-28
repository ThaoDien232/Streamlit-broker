import streamlit as st
import pandas as pd
import requests
from datetime import datetime
st.set_page_config(layout="wide")

# Load the prop book data with error handling
@st.cache_data
def load_data():
    try:
        return pd.read_excel("Prop book.xlsx")
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame()
df_book = load_data()
def sort_quarters_by_date(quarters):
    def key(q):
        try:
            q_num = int(q[0])
            year = int(q[2:])
            full_year = 2000 + year if year < 50 else 1900 + year
            return full_year * 10 + q_num
        except Exception:
            return q
    return sorted(quarters, key=key)

def fetch_historical_price(ticker: str) -> pd.DataFrame:
    """
    Fetch daily stock prices from TCBS API for the given ticker.
    Returns DataFrame with 'tradingDate', 'open', 'high', 'low', 'close', 'volume'.
    """
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",
        "from": "0",
        "to": str(int(datetime.now().timestamp()))
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and data['data']:
            df = pd.DataFrame(data['data'])
            # Convert tradingDate to datetime (ISO or ms)
            if 'tradingDate' in df.columns:
                if df['tradingDate'].dtype == 'object' and df['tradingDate'].str.contains('T').any():
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'])
                else:
                    df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms')
            # Only keep relevant columns
            keep = ['tradingDate', 'open', 'high', 'low', 'close', 'volume']
            return df[[col for col in keep if col in df.columns]]
        else:
            print("No data found in response")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_close_price(df: pd.DataFrame, target_date: str = None):
    """
    Get closing price on or before target_date.
    If target_date is None, get the latest available price.
    """
    if df.empty:
        return None
    if target_date:
        target = pd.to_datetime(target_date).tz_localize('UTC')
        df2 = df[df['tradingDate'] <= target]
        if df2.empty:
            return None
        return df2.iloc[-1]['close']
    else:
        return df.iloc[-1]['close']

def get_quarter_end_prices(tickers, quarter):
    q_map = {"1Q":"-03-31", "2Q":"-06-30", "3Q":"-09-30", "4Q":"-12-31"}
    q_part, y_part = quarter[:2], quarter[2:]
    date_str = f"{2000+int(y_part)}{q_map.get(q_part, '-12-31')}"
    prices = {}
    for ticker in tickers:
        if ticker.upper() == "OTHERS":
            prices[ticker] = None
        else:
            price_df = fetch_historical_price(ticker)
            prices[ticker] = get_close_price(price_df, date_str)
    return prices

def get_current_prices(tickers):
    prices = {}
    for ticker in tickers:
        if ticker.upper() == "OTHERS":
            prices[ticker] = 0
        else:
            price_df = fetch_historical_price(ticker)
            prices[ticker] = get_close_price(price_df)
    return prices

def calculate_profit_loss(df, quarter_prices, current_prices, quarter):
    """Calculate profit/loss from quarter-end to current prices"""
    df_calc = df.copy()
    # Add quarter-end price column
    df_calc['Quarter_End_Price'] = df_calc['Ticker'].map(quarter_prices)
    df_calc['Current_Price'] = df_calc['Ticker'].map(current_prices)
    # Calculate volume using quarter-end prices (volume at quarter end)
    df_calc['Volume'] = df_calc.apply(lambda row: 
        0 if row['Ticker'].upper() == 'OTHERS' or pd.isna(row['Quarter_End_Price']) or row['Quarter_End_Price'] == 0
        else row['FVTPL value'] / row['Quarter_End_Price'], axis=1)
    # Calculate quarter-end market value
    df_calc['Quarter_End_Market_Value'] = df_calc['Volume'] * df_calc['Quarter_End_Price'].fillna(0)
    # Calculate current market value using the same volume but current prices
    df_calc['Current_Market_Value'] = df_calc['Volume'] * df_calc['Current_Price'].fillna(0)
    # Calculate profit/loss from current market value minus FVTPL value
    df_calc['Profit_Loss'] = df_calc['Current_Market_Value'] - df_calc['FVTPL value']
    df_calc['Total_Profit_Loss'] = df_calc['Profit_Loss'].sum()
    df_calc['Profit_Loss_Pct'] = df_calc.apply(lambda row:
        0 if row['Quarter_End_Market_Value'] == 0 else (row['Profit_Loss'] / row['Quarter_End_Market_Value'] * 100), axis=1).round(1)
    return df_calc
    
def formatted_table(df, selected_quarters=None, key_suffix="", show_selectbox=True):
    if df.empty:
        return pd.DataFrame()
    
    # Numeric columns selection
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    import hashlib, time
    df_cols_str = ','.join(df.columns.astype(str))
    quarters_str = ','.join(selected_quarters) if selected_quarters is not None else ''
    broker_info = df['Broker'].iloc[0] if 'Broker' in df.columns and not df.empty else 'unknown'
    unique_str = f"{df_cols_str}-{quarters_str}-{broker_info}-{key_suffix}-{time.time()}"
    selectbox_key = f"value_col_{hashlib.md5(unique_str.encode()).hexdigest()}"
    
    # For download operations, skip the selectbox and use the first numeric column
    if not show_selectbox:
        value_col = numeric_cols[0] if numeric_cols else None
    else:
        value_col = numeric_cols[0] if len(numeric_cols) == 1 else st.selectbox("Select value column:", numeric_cols, key=selectbox_key)

    # Only show rows where the selected value column is present and the other is not
    other_col = None
    if value_col == 'FVTPL value' and 'AFS value' in df.columns:
        other_col = 'AFS value'
    elif value_col == 'AFS value' and 'FVTPL value' in df.columns:
        other_col = 'FVTPL value'
    if other_col:
        df = df[(df[other_col].isnull() | (df[other_col] == 0))]
    
    if selected_quarters is None:
        all_quarters = sort_quarters_by_date(df['Quarter'].unique())
    else:
        all_quarters = sort_quarters_by_date(selected_quarters)
    
    # Filter out 'PBT' for pivot table calculation
    df_no_pbt = df[df['Ticker'] != 'PBT']
    
    # Group and aggregate the data
    group_cols = ['Ticker', 'Quarter']
    df_no_pbt = df_no_pbt.groupby(group_cols, as_index=False).sum()
    
    # Create pivot table
    pivot_table = df_no_pbt.pivot(
        index='Ticker',
        columns='Quarter',
        values=value_col
    ).fillna(0)
    
    pivot_table = pivot_table.reindex(columns=all_quarters, fill_value=0)

    # Calculate Profit/Loss for each ticker
    tickers = [t for t in pivot_table.index if t.upper() not in ['OTHERS', 'PBT']]
    profit_dict, pct_dict = {}, {}
    
    # Determine if all tickers have zero for the latest quarter
    latest_quarter = all_quarters[-1]
    all_latest_zero = all(pivot_table.at[t, latest_quarter] == 0 for t in tickers)
    for t in tickers:
        # Find the latest quarter with a nonzero value for this ticker
        nonzero_quarters = [q for q in all_quarters if pivot_table.at[t, q] != 0]
        if not nonzero_quarters:
            profit_dict[t] = ''
            pct_dict[t] = ''
            continue
        # If all tickers have zero for the latest quarter, revert to previous quarter for all
        if all_latest_zero:
            q = nonzero_quarters[-1]
            q_price = get_quarter_end_prices([t], q)[t]
            c_price = get_current_prices([t])[t]
            val = pivot_table.at[t, q]
            if q_price and c_price and q_price != 0 and val != 0:
                vol = val / q_price
                p_start = vol * q_price
                p_now = vol * c_price
                profit = p_now - p_start
                pct = 0 if p_start == 0 else (profit / p_start * 100)
                profit_dict[t] = profit
                pct_dict[t] = pct
            else:
                profit_dict[t] = ''
                pct_dict[t] = ''
        else:
            # Only calculate for tickers with nonzero value in the latest quarter
            if pivot_table.at[t, latest_quarter] != 0:
                q = latest_quarter
                q_price = get_quarter_end_prices([t], q)[t]
                c_price = get_current_prices([t])[t]
                val = pivot_table.at[t, q]
                if q_price and c_price and q_price != 0 and val != 0:
                    vol = val / q_price
                    p_start = vol * q_price
                    p_now = vol * c_price
                    profit = p_now - p_start
                    pct = 0 if p_start == 0 else (profit / p_start * 100)
                    profit_dict[t] = profit
                    pct_dict[t] = pct
                else:
                    profit_dict[t] = ''
                    pct_dict[t] = ''
            else:
                # Do not calculate profit/loss for tickers with zero in the latest quarter
                profit_dict[t] = ''
                pct_dict[t] = ''
    
    # Add Profit/Loss and % Profit/Loss columns
    profit_col = "Profit/Loss since latest quarter"
    pct_col = "% Profit/Loss"
    pivot_table[profit_col] = pivot_table.index.map(lambda t: profit_dict.get(t, '') if t not in ['PBT'] else '')
    pivot_table[pct_col] = pivot_table.index.map(lambda t: pct_dict.get(t, '') if t not in ['PBT'] else '')

    # Add PBT and total rows separately
    pbt_rows = df[df['Ticker'] == 'PBT']
    if not pbt_rows.empty:
        pbt_pivot = pbt_rows.pivot_table(
            index='Ticker',
            columns='Quarter',
            values=value_col,
            aggfunc='first',  # Take the value as-is, no sum
            fill_value=0
        )
        pbt_pivot = pbt_pivot.reindex(columns=all_quarters, fill_value=0)
    
    rows = pivot_table.index.tolist()
    main_rows = pivot_table.drop([r for r in ['Others', 'PBT'] if r in rows])
    others_rows = pivot_table.loc[['Others']] if 'Others' in rows else pd.DataFrame()
    
    pivot_table_no_pbt = pd.concat([main_rows, others_rows]) if not others_rows.empty else main_rows
    
    # Total row calculation
    rows_for_total = [idx for idx in pivot_table_no_pbt.index if idx not in ['Total']]
    total_row = {}
    for col in pivot_table_no_pbt.columns:
        if "%" in str(col):
            total_row[col] = ""
        elif col == profit_col:
            total_row[col] = pd.to_numeric(pivot_table_no_pbt.loc[rows_for_total, col], errors='coerce').sum()
        else:
            total_row[col] = pd.to_numeric(pivot_table_no_pbt.loc[rows_for_total, col], errors='coerce').sum()
    
    total_df = pd.DataFrame([total_row], index=["Total"])
    pivot_table_final = pd.concat([pivot_table_no_pbt, total_df])
    
    if not pbt_rows.empty:
        pivot_table_final = pd.concat([pivot_table_final, pbt_pivot])
    
    # Formatting numbers
    formatted_table = pivot_table_final.copy()
    import numpy as np
    for col in formatted_table.columns:
        if "%" in str(col):
            formatted_table[col] = formatted_table[col].apply(
                lambda x: f"{x:,.1f}%" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notnull(x) else ""
            )
        elif "Profit/Loss" in str(col):
            formatted_table[col] = formatted_table[col].apply(
                lambda x: f"{x:,.1f}" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notnull(x) else ""
            )
        else:
            formatted_table[col] = formatted_table[col].apply(
                lambda x: f"{x:,.1f}" if isinstance(x, (int, float, np.integer, np.floating)) and pd.notnull(x) else ""
            )

    return formatted_table

st.title("Prop Book Dashboard")

# Add refresh and export buttons
col1, col2, col3 = st.columns([2, 1, 1])
with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

def display_prop_book_table():
    """Display prop book data by broker and quarter"""
    
    brokers = sorted(df_book['Broker'].unique())
    quarters = sort_quarters_by_date(df_book['Quarter'].unique())

    # --- Disclaimers dictionary ---
    disclaimers = {
        "VIX": "Calculated profit/loss may not be correct from lack of latest holding",
        "VCI": "Prop trade is held in AFS, not marked-to-market",
        "HCM": "This is not HCM's prop book, not disclosed"
    }


    selected_brokers = st.selectbox(
        "Select Brokers:",
        options=brokers,
        index=0
    )

    if selected_brokers in disclaimers:
        st.warning(disclaimers[selected_brokers])
    selected_quarters = st.multiselect(
        "Select Quarters:",
        options=quarters,
        default=quarters
    )

    # Export button in top right - move this logic to col3
    with col3:
        if selected_quarters:
            # Create data organized by broker for the selected quarters
            combined_csv = ""
            brokers_for_export = sorted(df_book['Broker'].unique())
            
            for broker in brokers_for_export:
                broker_df = df_book[(df_book['Broker'] == broker) & (df_book['Quarter'].isin(selected_quarters))].copy()
                if not broker_df.empty:
                    combined_csv += f"\n{broker} Prop Book\n"
                    
                    # Get FVTPL data
                    fvtpl_df = broker_df[(broker_df['FVTPL value'].notnull() & (broker_df['FVTPL value'] != 0))].copy()
                    if not fvtpl_df.empty:
                        fvtpl_data = formatted_table(fvtpl_df, selected_quarters, key_suffix=f"export_fvtpl_{broker}", show_selectbox=False)
                        combined_csv += f"\nFVTPL Values:\n"
                        combined_csv += fvtpl_data.to_csv(index=True)
                        combined_csv += "\n"
                    
                    # Get AFS data if AFS column exists
                    if 'AFS value' in broker_df.columns:
                        afs_df = broker_df[(broker_df['AFS value'].notnull() & (broker_df['AFS value'] != 0))].copy()
                        if not afs_df.empty:
                            # Copy AFS to FVTPL column for processing since formatted_table expects FVTPL
                            afs_df_modified = afs_df.copy()
                            afs_df_modified['FVTPL value'] = afs_df_modified['AFS value']
                            afs_data = formatted_table(afs_df_modified, selected_quarters, key_suffix=f"export_afs_{broker}", show_selectbox=False)
                            combined_csv += f"AFS Values:\n"
                            combined_csv += afs_data.to_csv(index=True)
                            combined_csv += "\n"
                    
                    combined_csv += "\n"  # Extra spacing between brokers
            
            st.download_button(
                label="Export Data",
                data=combined_csv,
                file_name=f"prop_book_all_brokers_{'_'.join(selected_quarters)}.csv",
                mime="text/csv"
            )

    filtered_df = df_book.copy()
    # Only include PBT rows for the selected broker and selected quarters
    if selected_brokers and 'Broker' in df_book.columns:
        filtered_df = filtered_df[(filtered_df['Broker'] == selected_brokers) | ((filtered_df['Ticker'] == 'PBT') & (filtered_df['Broker'] == selected_brokers))]
    if selected_quarters and 'Quarter' in df_book.columns:
        filtered_df = filtered_df[filtered_df['Quarter'].isin(selected_quarters)]

    # Get the latest quarter chronologically with data for the selected broker or PBT ---
    available_quarters = sort_quarters_by_date(filtered_df['Quarter'].unique())
    latest_quarter = available_quarters[-1] if available_quarters else None

    # Display the prop book table with additional columns
    st.subheader(f"{selected_brokers} Prop Book")

    with st.spinner("Loading data and calculating price changes..."):
        # Use available_quarters for both display and calculation
        formatted_df = formatted_table(filtered_df, available_quarters, key_suffix=f"display_{selected_brokers}")
        st.dataframe(formatted_df, use_container_width=True)

# Main application
def main():
    display_prop_book_table()

if __name__ == "__main__":
    main ()