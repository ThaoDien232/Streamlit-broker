"""
Valuation analysis utilities for brokerage sector.
Adapted from banking sector valuation analysis with broker-specific modifications.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def get_metric_column(metric_type: str) -> str:
    """Map UI metric labels to dataframe column names."""
    mapping = {
        "ROE": "ROE",
        "ROA": "ROA",
        "P/E": "PE",
        "P/B": "PB",
        "NPAT": "NPAT",
        "Total Equity": "TOTAL_EQUITY",
        "Total Assets": "TOTAL_ASSETS"
    }
    return mapping.get(metric_type, metric_type)

def remove_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Remove outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def calculate_distribution_stats(df: pd.DataFrame, ticker: str, metric_col: str) -> Dict:
    """
    Calculate distribution statistics for candle chart
    Returns percentiles and current value
    """
    # Get historical data for the ticker
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()

    if len(ticker_data) < 10:  # Reduced minimum for broker data
        return None

    # Remove outliers for cleaner distribution
    clean_data = remove_outliers_iqr(ticker_data, multiplier=2.0)

    if len(clean_data) < 5:  # Ensure we still have some data after cleaning
        clean_data = ticker_data

    # Get current value (most recent)
    current_value = df[df['TICKER'] == ticker][metric_col].iloc[-1] if len(df[df['TICKER'] == ticker]) > 0 else None

    # Calculate percentiles
    stats_dict = {
        'p5': clean_data.quantile(0.05),
        'p25': clean_data.quantile(0.25),
        'p50': clean_data.quantile(0.50),  # Median
        'p75': clean_data.quantile(0.75),
        'p95': clean_data.quantile(0.95),
        'current': current_value,
        'count': len(clean_data)
    }

    # Calculate current value percentile
    if current_value is not None and not pd.isna(current_value):
        percentile_rank = (clean_data <= current_value).sum() / len(clean_data) * 100
        stats_dict['percentile'] = percentile_rank
    else:
        stats_dict['percentile'] = None

    return stats_dict

def calculate_historical_stats(df: pd.DataFrame, ticker: str, metric_col: str) -> Dict:
    """
    Calculate historical statistics for time series chart
    Returns mean, std dev, and current z-score
    """
    # Get historical data
    ticker_data = df[df['TICKER'] == ticker][[metric_col, 'TRADE_DATE']].copy()
    ticker_data = ticker_data.dropna()

    if len(ticker_data) < 8:  # Reduced minimum for broker data
        return None

    # Sort by date
    ticker_data = ticker_data.sort_values('TRADE_DATE')

    # Remove outliers
    clean_values = remove_outliers_iqr(ticker_data[metric_col], multiplier=3.0)

    if len(clean_values) < 5:  # Ensure we have enough clean data
        clean_values = ticker_data[metric_col]

    # Calculate statistics
    mean_val = clean_values.mean()
    std_val = clean_values.std()
    current_val = ticker_data[metric_col].iloc[-1] if len(ticker_data) > 0 else None

    stats_dict = {
        'mean': mean_val,
        'std': std_val,
        'upper_1sd': mean_val + std_val,
        'lower_1sd': mean_val - std_val,
        'upper_2sd': mean_val + 2 * std_val,
        'lower_2sd': mean_val - 2 * std_val,
        'current': current_val
    }

    # Calculate z-score
    if current_val is not None and not pd.isna(current_val) and std_val > 0:
        z_score = (current_val - mean_val) / std_val
        stats_dict['z_score'] = z_score
    else:
        stats_dict['z_score'] = None

    return stats_dict

def calculate_cdf(df: pd.DataFrame, ticker: str, metric_col: str) -> float:
    """
    Calculate cumulative distribution function (percentile rank)
    Returns value between 0 and 100
    """
    # Get historical data
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()

    if len(ticker_data) < 8:  # Reduced minimum for broker data
        return None

    # Get current value
    current_value = ticker_data.iloc[-1] if len(ticker_data) > 0 else None

    if current_value is None or pd.isna(current_value):
        return None

    # Remove outliers
    clean_data = remove_outliers_iqr(ticker_data, multiplier=3.0)

    if len(clean_data) < 5:
        clean_data = ticker_data

    # Calculate CDF (percentile)
    cdf_value = (clean_data <= current_value).sum() / len(clean_data) * 100

    return cdf_value

def get_valuation_status(z_score: float, metric_type: str = "ROE") -> Tuple[str, str]:
    """
    Get valuation status based on z-score
    Returns (status, color)

    For metrics like ROE/ROA: Higher values are better
    For metrics like P/E/P/B: Lower values are better (would need to invert logic)
    """
    if z_score is None:
        return ("N/A", "gray")

    # For ROE/ROA and similar performance metrics (higher is better)
    if metric_type.upper() in ['ROE', 'ROA', 'NPAT']:
        if z_score > 1.5:
            return ("Excellent", "darkgreen")
        elif z_score > 0.5:
            return ("Good", "green")
        elif z_score > -0.5:
            return ("Average", "yellow")
        elif z_score > -1.5:
            return ("Below Average", "orange")
        else:
            return ("Poor", "red")

    # For valuation ratios like P/E, P/B (lower is typically better)
    elif metric_type.upper() in ['PE', 'PB', 'P/E', 'P/B']:
        if z_score < -1.5:
            return ("Very Cheap", "darkgreen")
        elif z_score < -0.5:
            return ("Cheap", "green")
        elif z_score < 0.5:
            return ("Fair", "yellow")
        elif z_score < 1.5:
            return ("Expensive", "orange")
        else:
            return ("Very Expensive", "red")

    # Default case
    else:
        if z_score > 1.5:
            return ("Very High", "darkgreen")
        elif z_score > 0.5:
            return ("High", "green")
        elif z_score > -0.5:
            return ("Average", "yellow")
        elif z_score > -1.5:
            return ("Low", "orange")
        else:
            return ("Very Low", "red")

def prepare_statistics_table(df: pd.DataFrame, metric_col: str, metric_type: str = "ROE") -> pd.DataFrame:
    """
    Prepare statistics table with all tickers
    """
    results = []

    # Get unique tickers
    tickers = df['TICKER'].unique()

    for ticker in tickers:
        # Get ticker type
        ticker_type = df[df['TICKER'] == ticker]['Type'].iloc[0] if len(df[df['TICKER'] == ticker]) > 0 else "Unknown"

        # Skip if insufficient data
        ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()
        if len(ticker_data) < 5:  # Reduced minimum for broker data
            continue

        # Calculate statistics
        hist_stats = calculate_historical_stats(df, ticker, metric_col)
        if hist_stats is None:
            continue

        cdf_value = calculate_cdf(df, ticker, metric_col)

        # Get status
        status, color = get_valuation_status(hist_stats.get('z_score'), metric_type)

        # Determine if this is a sector aggregate
        is_sector = ticker in ['Sector', 'Listed', 'Unlisted', 'All_Brokers']

        results.append({
            'Ticker': ticker,
            'Type': ticker_type,
            'Current': hist_stats.get('current', None),
            'Mean': hist_stats.get('mean', None),
            'CDF (%)': cdf_value,
            'Z-Score': hist_stats.get('z_score', None),
            'Status': status,
            'IsSector': is_sector
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Sort properly: Sector first, then Listed, then Unlisted, then individual brokers
    if not results_df.empty:
        # Separate sectors and individual brokers
        sectors_df = results_df[results_df['IsSector'] == True].copy()
        brokers_df = results_df[results_df['IsSector'] == False].copy()

        # Build the final dataframe in the correct order
        final_rows = []

        # First add "Sector" if it exists
        if 'Sector' in sectors_df['Ticker'].values:
            final_rows.append(sectors_df[sectors_df['Ticker'] == 'Sector'])

        # Add sector aggregates in order
        for sector_type in ['Listed', 'Unlisted', 'All_Brokers']:
            if sector_type in sectors_df['Ticker'].values:
                final_rows.append(sectors_df[sectors_df['Ticker'] == sector_type])

        # Add individual brokers by type
        for broker_type in ['Listed', 'Unlisted']:
            component_brokers = brokers_df[brokers_df['Type'] == broker_type]
            if not component_brokers.empty:
                # Sort brokers by current value (descending for performance metrics)
                component_brokers = component_brokers.sort_values('Current', ascending=False)
                final_rows.append(component_brokers)

        # Add any remaining brokers that don't belong to standard types
        other_brokers = brokers_df[~brokers_df['Type'].isin(['Listed', 'Unlisted'])]
        if not other_brokers.empty:
            other_brokers = other_brokers.sort_values('Current', ascending=False)
            final_rows.append(other_brokers)

        # Combine all rows
        if final_rows:
            results_df = pd.concat(final_rows, ignore_index=True)

        # Drop helper columns
        results_df = results_df.drop(['IsSector'], axis=1, errors='ignore')

    return results_df

def get_sector_and_components(df: pd.DataFrame, sector: str, include_unlisted: bool = True) -> List[str]:
    """
    Get list of tickers for a sector and its components
    """
    if sector == "All_Brokers":
        # Return all brokers
        listed_brokers = sorted(df[df['Type'] == 'Listed']['TICKER'].unique().tolist())
        unlisted_brokers = sorted(df[df['Type'] == 'Unlisted']['TICKER'].unique().tolist()) if include_unlisted else []
        return ['All_Brokers'] + listed_brokers + unlisted_brokers
    elif sector == "Listed":
        # Return listed brokers only
        listed_brokers = sorted(df[df['Type'] == 'Listed']['TICKER'].unique().tolist())
        return ['Listed'] + listed_brokers
    elif sector == "Unlisted":
        # Return unlisted brokers only
        unlisted_brokers = sorted(df[df['Type'] == 'Unlisted']['TICKER'].unique().tolist())
        return ['Unlisted'] + unlisted_brokers
    else:
        # Return sector aggregate plus all component brokers
        all_tickers = df['TICKER'].unique().tolist()
        # Filter based on include_unlisted flag
        if not include_unlisted:
            all_tickers = [t for t in all_tickers if len(str(t)) == 3 and str(t).isalpha()]
        return ['All_Brokers'] + sorted([t for t in all_tickers if t not in ['Sector', 'Listed', 'Unlisted', 'All_Brokers']])

def generate_valuation_histogram(df: pd.DataFrame, ticker: str, metric_col: str, n_bins: int = 8) -> Dict:
    """
    Generate histogram data for a ticker's valuation metric
    Returns bin edges, counts, current value bin, and formatted data for visualization
    """
    # Get historical data for the ticker
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()

    if len(ticker_data) < 8:  # Reduced minimum for broker data
        return None

    # Remove outliers for cleaner distribution
    clean_data = remove_outliers_iqr(ticker_data, multiplier=3.0)

    if len(clean_data) < 5:
        clean_data = ticker_data

    # Get current value (most recent)
    current_value = ticker_data.iloc[-1] if len(ticker_data) > 0 else None

    if current_value is None or pd.isna(current_value):
        return None

    # Create histogram bins
    counts, bin_edges = np.histogram(clean_data, bins=min(n_bins, len(clean_data)//2))

    # Find which bin the current value belongs to
    current_bin_idx = None
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= current_value < bin_edges[i + 1]:
            current_bin_idx = i
            break
    # Handle edge case where current value equals the last edge
    if current_value == bin_edges[-1] and len(counts) > 0:
        current_bin_idx = len(counts) - 1

    # Create bin centers for plotting
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    # Format bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        label = f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}"
        bin_labels.append(label)

    # Calculate percentile
    percentile = (clean_data <= current_value).sum() / len(clean_data) * 100

    histogram_data = {
        'ticker': ticker,
        'current_value': current_value,
        'current_bin_idx': current_bin_idx,
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers,
        'bin_labels': bin_labels,
        'counts': counts.tolist(),
        'percentile': percentile,
        'n_total': len(clean_data),
        'min_value': float(clean_data.min()),
        'max_value': float(clean_data.max()),
        'median': float(clean_data.median())
    }

    return histogram_data

def create_sector_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sector aggregates for Listed and Unlisted brokers
    """
    try:
        base = df.copy()

        # Get only individual brokers (not existing aggregates)
        individual_brokers = base[~base['TICKER'].isin(['Sector', 'Listed', 'Unlisted', 'All_Brokers'])]

        if individual_brokers.empty:
            return df

        # Ensure needed columns exist
        needed_cols = {'TRADE_DATE', 'TICKER', 'Type'}
        value_cols = [col for col in df.columns if col not in ['TICKER', 'TRADE_DATE', 'Type']]

        if not needed_cols.issubset(set(individual_brokers.columns)):
            return df

        # Create aggregates for each date and type
        agg_data = []

        # Listed brokers aggregate
        listed_data = individual_brokers[individual_brokers['Type'] == 'Listed']
        if not listed_data.empty:
            listed_agg = (
                listed_data
                .groupby(['TRADE_DATE'])[value_cols]
                .median()
                .reset_index()
            )
            listed_agg['TICKER'] = 'Listed'
            listed_agg['Type'] = 'Listed'
            agg_data.append(listed_agg)

        # Unlisted brokers aggregate
        unlisted_data = individual_brokers[individual_brokers['Type'] == 'Unlisted']
        if not unlisted_data.empty:
            unlisted_agg = (
                unlisted_data
                .groupby(['TRADE_DATE'])[value_cols]
                .median()
                .reset_index()
            )
            unlisted_agg['TICKER'] = 'Unlisted'
            unlisted_agg['Type'] = 'Unlisted'
            agg_data.append(unlisted_agg)

        # All brokers aggregate
        all_agg = (
            individual_brokers
            .groupby(['TRADE_DATE'])[value_cols]
            .median()
            .reset_index()
        )
        all_agg['TICKER'] = 'All_Brokers'
        all_agg['Type'] = 'All_Brokers'
        agg_data.append(all_agg)

        # Combine with original data
        if agg_data:
            combined_df = pd.concat([base] + agg_data, ignore_index=True, sort=False)
            # Remove any duplicate combinations of TICKER and TRADE_DATE
            combined_df = combined_df.drop_duplicates(subset=['TICKER', 'TRADE_DATE'])
            return combined_df

        return df

    except Exception as e:
        print(f"Could not compute sector aggregates: {e}")
        return df