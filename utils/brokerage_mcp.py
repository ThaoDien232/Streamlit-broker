"""Brokerage MCP tool registry and tool implementations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import brokerage_data
from utils.db import run_query
from utils.openai_commentary import generate_commentary
from utils.valuation_data import load_brokerage_valuation_data
from utils.valuation_analysis import calculate_distribution_stats, get_metric_column


def _now_ts() -> float:
    return time.time()


def _serialize_dataframe(df: pd.DataFrame, max_rows: int = 2000) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    limited = df.head(max_rows).copy()
    for col in limited.columns:
        if pd.api.types.is_datetime64_any_dtype(limited[col]):
            limited[col] = limited[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_numeric_dtype(limited[col]):
            limited[col] = limited[col].astype(float)
    return limited.to_dict(orient="records")


def _normalize_ticker_list(tickers: Optional[List[str]]) -> List[str]:
    if not tickers:
        try:
            return brokerage_data.get_available_tickers()
        except Exception:
            return []
    return sorted({t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()})


def _quarter_label(year: int, quarter: int) -> str:
    short_year = str(year)[-2:]
    return f"{quarter}Q{short_year}"


def _parse_period_label(label: str) -> Tuple[Optional[int], Optional[int]]:
    if not isinstance(label, str):
        return None, None
    text = label.strip().upper()
    if text.endswith("-YTD"):
        try:
            year = int(text.split("-")[0])
            return year, None
        except ValueError:
            return None, None
    if "Q" in text:
        if "-" in text:
            # Format like 2024-Q3
            try:
                year_part, quarter_part = text.split("-")
                quarter = int(quarter_part.replace("Q", ""))
                year = int(year_part)
                return year, quarter
            except ValueError:
                return None, None
        else:
            # Format like 1Q24
            try:
                quarter = int(text.split("Q")[0])
                year_suffix = text.split("Q")[1]
                year = int(year_suffix)
                if year < 100:
                    year = 2000 + year if year <= 50 else 1900 + year
                return year, quarter
            except (ValueError, IndexError):
                return None, None
    try:
        year = int(text)
        return year, None
    except ValueError:
        return None, None


def _build_quarter_label(year: int, quarter: int) -> str:
    return f"{quarter}Q{str(year)[-2:]}"


def _metric_group_mapping() -> Dict[str, List[str]]:
    return {
        "revenue": [
            "Net_Brokerage_Income",
            "Net_IB_Income",
            "Net_investment_income",
            "Net_Margin_lending_Income",
            "Net_other_operating_income",
            "Total_Operating_Income",
        ],
        "profitability": ["PBT", "NPAT", "ROE", "ROA", "CIR"],
        "balance_sheet": [
            "Total_Assets",
            "Total_Equity",
            "Borrowing_Balance",
            "Margin_Lending_book",
        ],
        "costs": ["SG_A", "Interest_Expense"],
        "market": ["NET_BROKERAGE_FEE", "MARGIN_EQUITY_RATIO", "MARGIN_LENDING_RATE"],
    }


@dataclass
class CachedResult:
    timestamp: float
    payload: Dict[str, Any]


class BrokerageMCP:
    """Lightweight MCP-style tool registry for the brokerage chatbot."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = ttl_seconds
        self._tools: Dict[str, Callable[..., Dict[str, Any]]] = {}
        self.tool_specs: List[Dict[str, Any]] = []
        self._cache: Dict[str, CachedResult] = {}
        self._register_tools()

    def _cache_key(self, name: str, arguments: Dict[str, Any]) -> str:
        serialized = json.dumps(arguments, sort_keys=True, default=str)
        return f"{name}:{serialized}"

    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        cached = self._cache.get(key)
        if not cached:
            return None
        if _now_ts() - cached.timestamp > self.ttl:
            del self._cache[key]
            return None
        return cached.payload

    def _set_cache(self, key: str, payload: Dict[str, Any]) -> None:
        self._cache[key] = CachedResult(timestamp=_now_ts(), payload=payload)

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            return {"status": "failed", "error": f"Unknown tool '{name}'"}

        cache_key = self._cache_key(name, arguments)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return {"status": "success", "cached": True, "data": cached}

        try:
            result = tool(**arguments)
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}

        if isinstance(result, dict):
            self._set_cache(cache_key, result)
            return {"status": "success", "cached": False, "data": result}

        payload = {"result": result}
        self._set_cache(cache_key, payload)
        return {"status": "success", "cached": False, "data": payload}

    def _register_tool(self, name: str, description: str, parameters: Dict[str, Any]):
        def decorator(func: Callable[..., Dict[str, Any]]):
            cleaned_properties: Dict[str, Any] = {}
            required_fields: List[str] = []
            for key, schema in parameters.items():
                schema_copy = dict(schema)
                if schema_copy.pop("required", False):
                    required_fields.append(key)
                cleaned_properties[key] = schema_copy

            param_schema: Dict[str, Any] = {
                "type": "object",
                "properties": cleaned_properties,
            }
            if required_fields:
                param_schema["required"] = required_fields

            self._tools[name] = func
            self.tool_specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": param_schema,
                    },
                }
            )
            return func

        return decorator

    def _register_tools(self) -> None:
        self._register_data_availability()
        self._register_broker_info()
        self._register_query_historical()
        self._register_query_forecast()
        self._register_valuation()
        self._register_stock_performance()
        self._register_commentary()

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _register_data_availability(self) -> None:
        @self._register_tool(
            name="get_data_availability",
            description="Return brokerage data coverage including latest quarter, available years, and forecast horizon.",
            parameters={},
        )
        def _tool() -> Dict[str, Any]:
            latest_actual_query = """
                SELECT TOP 1 YEARREPORT, LENGTHREPORT, QUARTER_LABEL
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 1 AND LENGTHREPORT BETWEEN 1 AND 4
                ORDER BY YEARREPORT DESC, LENGTHREPORT DESC
            """
            annual_years_query = """
                SELECT DISTINCT YEARREPORT
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 1 AND LENGTHREPORT = 5
                ORDER BY YEARREPORT DESC
            """
            recent_quarters_query = """
                SELECT DISTINCT TOP 8 QUARTER_LABEL
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 1 AND LENGTHREPORT BETWEEN 1 AND 4 AND QUARTER_LABEL IS NOT NULL
                ORDER BY YEARREPORT DESC, LENGTHREPORT DESC
            """
            recent_years_query = """
                SELECT DISTINCT TOP 5 YEARREPORT
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 1 AND LENGTHREPORT = 5
                ORDER BY YEARREPORT DESC
            """
            forecast_years_query = """
                SELECT DISTINCT YEARREPORT
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 0
                ORDER BY YEARREPORT
            """

            latest_per_ticker_query = """
                WITH ranked AS (
                    SELECT
                        TICKER,
                        YEARREPORT,
                        LENGTHREPORT,
                        QUARTER_LABEL,
                        ROW_NUMBER() OVER (PARTITION BY TICKER ORDER BY YEARREPORT DESC, LENGTHREPORT DESC) AS rn
                    FROM dbo.BrokerageMetrics
                    WHERE ACTUAL = 1 AND LENGTHREPORT BETWEEN 1 AND 4
                      AND TICKER NOT IN ({excluded})
                )
                SELECT TICKER, YEARREPORT, LENGTHREPORT, QUARTER_LABEL
                FROM ranked
                WHERE rn = 1
            """

            latest = run_query(latest_actual_query)
            annual_years = run_query(annual_years_query)
            recent_quarters_df = run_query(recent_quarters_query)
            recent_years_df = run_query(recent_years_query)
            forecast_years = run_query(forecast_years_query)

            excluded_list = ','.join([f"'{t}'" for t in brokerage_data.EXCLUDED_TICKERS])
            per_ticker_df = run_query(latest_per_ticker_query.format(excluded=excluded_list))

            latest_period = None
            if not latest.empty:
                row = latest.iloc[0]
                latest_period = {
                    "year": int(row["YEARREPORT"]),
                    "quarter": int(row["LENGTHREPORT"]),
                    "label": row.get("QUARTER_LABEL"),
                }

            recent_quarters = []
            if not recent_quarters_df.empty:
                recent_quarters = list(recent_quarters_df["QUARTER_LABEL"].astype(str))

            recent_years = []
            if not recent_years_df.empty:
                recent_years = [str(int(year)) for year in recent_years_df["YEARREPORT"]]

            data = {
                "status": "success",
                "current_date": datetime.utcnow().strftime("%Y-%m-%d"),
                "latest_quarter": latest_period,
                "latest_year": recent_years[0] if recent_years else None,
                "recent_quarters": recent_quarters,
                "recent_years": recent_years,
                "available_annual_years": annual_years["YEARREPORT"].fillna(0).astype(int).tolist()
                if not annual_years.empty
                else [],
                "forecast_years": forecast_years["YEARREPORT"].fillna(0).astype(int).tolist()
                if not forecast_years.empty
                else [],
                "latest_quarters_by_ticker": []
                if per_ticker_df.empty
                else [
                    {
                        "ticker": row["TICKER"],
                        "year": int(row["YEARREPORT"]),
                        "quarter": int(row["LENGTHREPORT"]),
                        "label": row.get("QUARTER_LABEL"),
                    }
                    for _, row in per_ticker_df.iterrows()
                ],
            }

            return data

    def _register_broker_info(self) -> None:
        @self._register_tool(
            name="get_broker_info",
            description="List available brokerage tickers or provide details for specific tickers.",
            parameters={
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional array of broker tickers.",
                    "required": False,
                }
            },
        )
        def _tool(tickers: Optional[List[str]] = None) -> Dict[str, Any]:
            available = brokerage_data.get_available_tickers()
            if not tickers:
                return {"tickers": available, "count": len(available)}

            requested = _normalize_ticker_list(tickers)
            info: Dict[str, Any] = {}

            df = pd.DataFrame()
            if requested:
                placeholders = ",".join([f"'{t}'" for t in requested])
                metadata_query = f"""
                    SELECT TICKER, MIN(YEARREPORT) AS first_year, MAX(YEARREPORT) AS last_year
                    FROM dbo.BrokerageMetrics
                    WHERE TICKER IN ({placeholders}) AND ACTUAL = 1
                    GROUP BY TICKER
                """
                df = run_query(metadata_query)
            for ticker in requested:
                details = {
                    "available": ticker in available,
                }
                if not df.empty:
                    row = df[df["TICKER"] == ticker]
                    if not row.empty:
                        details.update(
                            {
                                "first_year": int(row.iloc[0]["first_year"]),
                                "last_year": int(row.iloc[0]["last_year"]),
                            }
                        )
                info[ticker] = details

            return {"requested": requested, "details": info, "available_universe": available}

    def _register_query_historical(self) -> None:
        parameters = {
            "frequency": {
                "type": "string",
                "enum": ["quarterly", "annual"],
                "description": "Data frequency.",
                "required": True,
            },
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of broker tickers.",
                "required": False,
            },
            "period": {
                "type": "string",
                "description": "Single period (e.g., 2024-Q3 or 1Q24 or 2024).",
                "required": False,
            },
            "periods": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple periods.",
                "required": False,
            },
            "metric": {
                "type": "string",
                "description": "Single metric KEYCODE.",
                "required": False,
            },
            "metric_group": {
                "type": "string",
                "description": "Metric group shortcut (revenue, profitability, balance_sheet, costs, market).",
                "required": False,
            },
        }

        @self._register_tool(
            name="query_historical_data",
            description="Retrieve brokerage historical metrics with optional filters.",
            parameters=parameters,
        )
        def _tool(
            frequency: str,
            tickers: Optional[List[str]] = None,
            period: Optional[str] = None,
            periods: Optional[List[str]] = None,
            metric: Optional[str] = None,
            metric_group: Optional[str] = None,
        ) -> Dict[str, Any]:
            frequency = frequency.lower()
            if frequency not in {"quarterly", "annual"}:
                raise ValueError("Frequency must be 'quarterly' or 'annual'.")

            ticker_list = _normalize_ticker_list(tickers)
            if not ticker_list:
                raise ValueError("No tickers provided and none available.")

            metrics: List[str] = []
            if metric:
                metrics = [metric]
            elif metric_group:
                mapping = _metric_group_mapping()
                metrics = mapping.get(metric_group.lower(), [])
            if not metrics:
                metrics = [
                    "Total_Operating_Income",
                    "Net_Brokerage_Income",
                    "Net_IB_Income",
                    "Net_investment_income",
                    "Net_Margin_lending_Income",
                    "PBT",
                    "NPAT",
                    "ROE",
                    "ROA",
                ]

            period_labels: List[str] = []
            total_periods = periods or []
            if period:
                total_periods.append(period)

            for entry in total_periods:
                year, quarter = _parse_period_label(entry)
                if year and quarter:
                    period_labels.append(_build_quarter_label(year, quarter))
                elif year and quarter is None and frequency == "annual":
                    period_labels.append(str(year))

            if not period_labels:
                # If no periods specified fetch last four quarters or last five years
                if frequency == "quarterly":
                    default_query = """
                        SELECT DISTINCT TOP 4 QUARTER_LABEL
                        FROM dbo.BrokerageMetrics
                        WHERE ACTUAL = 1 AND LENGTHREPORT BETWEEN 1 AND 4 AND QUARTER_LABEL IS NOT NULL
                        ORDER BY YEARREPORT DESC, LENGTHREPORT DESC
                    """
                else:
                    default_query = """
                        SELECT DISTINCT TOP 5 QUARTER_LABEL
                        FROM dbo.BrokerageMetrics
                        WHERE ACTUAL = 1 AND LENGTHREPORT = 5 AND QUARTER_LABEL IS NOT NULL
                        ORDER BY YEARREPORT DESC
                    """
                default_df = run_query(default_query)
                period_labels = default_df["QUARTER_LABEL"].tolist() if not default_df.empty else []

            if not period_labels:
                raise ValueError("No periods available for the requested dataset.")

            if frequency == "quarterly":
                length_condition = "LENGTHREPORT BETWEEN 1 AND 4"
            else:
                length_condition = "LENGTHREPORT = 5"

            placeholders_metrics = ",".join([f"'{m}'" for m in metrics])
            placeholders_periods = ",".join([f"'{p}'" for p in period_labels])
            placeholders_tickers = ",".join([f"'{t}'" for t in ticker_list])

            query = f"""
                SELECT TICKER, YEARREPORT, LENGTHREPORT, QUARTER_LABEL, KEYCODE, KEYCODE_NAME, VALUE
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 1
                  AND {length_condition}
                  AND KEYCODE IN ({placeholders_metrics})
                  AND QUARTER_LABEL IN ({placeholders_periods})
                  AND TICKER IN ({placeholders_tickers})
                ORDER BY TICKER, YEARREPORT, LENGTHREPORT, KEYCODE
            """

            df = run_query(query)
            return {"rows": _serialize_dataframe(df)}

    def _register_query_forecast(self) -> None:
        parameters = {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of broker tickers.",
                "required": False,
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of forecast KEYCODEs.",
                "required": False,
            },
        }

        @self._register_tool(
            name="query_forecast_data",
            description="Retrieve brokerage forecast metrics from database.",
            parameters=parameters,
        )
        def _tool(
            tickers: Optional[List[str]] = None,
            metrics: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            ticker_list = _normalize_ticker_list(tickers)
            if not ticker_list:
                raise ValueError("No tickers provided and none available.")

            metric_list = metrics or [
                "Net_Brokerage_Income",
                "Net_Margin_lending_Income",
                "Net_investment_income",
                "Net_IB_Income",
                "PBT",
                "NPAT",
            ]

            placeholders_metrics = ",".join([f"'{m}'" for m in metric_list])
            placeholders_tickers = ",".join([f"'{t}'" for t in ticker_list])

            query = f"""
                SELECT TICKER, YEARREPORT, LENGTHREPORT, QUARTER_LABEL, KEYCODE, KEYCODE_NAME, VALUE
                FROM dbo.BrokerageMetrics
                WHERE ACTUAL = 0
                  AND KEYCODE IN ({placeholders_metrics})
                  AND TICKER IN ({placeholders_tickers})
                ORDER BY TICKER, YEARREPORT
            """

            df = run_query(query)
            return {"rows": _serialize_dataframe(df)}

    def _register_valuation(self) -> None:
        parameters = {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of broker tickers.",
                "required": False,
            },
            "metric": {
                "type": "string",
                "description": "Valuation metric (PE, PB, ROE, ROA).",
                "required": False,
            },
            "years": {
                "type": "integer",
                "description": "Number of trailing years to include (default 5).",
                "required": False,
            },
        }

        @self._register_tool(
            name="get_valuation_analysis",
            description="Perform valuation distribution analysis for brokers.",
            parameters=parameters,
        )
        def _tool(
            tickers: Optional[List[str]] = None,
            metric: Optional[str] = None,
            years: Optional[int] = None,
        ) -> Dict[str, Any]:
            metric_label = metric or "PB"
            metric_col = get_metric_column(metric_label.upper())
            years = years or 5

            df = load_brokerage_valuation_data(years=years)
            if df.empty or metric_col not in df.columns:
                raise ValueError(f"Metric {metric_label} unavailable in valuation dataset.")

            ticker_list = _normalize_ticker_list(tickers)
            if not ticker_list:
                ticker_list = sorted(df["TICKER"].unique())

            results = []
            for ticker in ticker_list:
                stats = calculate_distribution_stats(df, ticker, metric_col)
                if not stats:
                    continue
                stats.update({"ticker": ticker})
                results.append(stats)

            return {"metric": metric_label, "tickers": ticker_list, "statistics": results}

    def _register_stock_performance(self) -> None:
        parameters = {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock tickers.",
                "required": True,
            },
            "start_date": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD).",
                "required": True,
            },
            "end_date": {
                "type": "string",
                "description": "End date (YYYY-MM-DD).",
                "required": True,
            },
        }

        @self._register_tool(
            name="get_stock_performance",
            description="Calculate price performance between two dates using TCBS API.",
            parameters=parameters,
        )
        def _tool(
            tickers: List[str],
            start_date: str,
            end_date: str,
        ) -> Dict[str, Any]:
            import requests

            def fetch_price(ticker: str, target: Optional[str]) -> Optional[float]:
                base_url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
                params = {
                    "ticker": ticker,
                    "type": "stock",
                    "resolution": "D",
                    "from": "0",
                    "to": str(int(time.time())),
                }
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "application/json",
                }

                resp = requests.get(base_url, params=params, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                if not data:
                    return None
                df = pd.DataFrame(data)
                df["tradingDate"] = pd.to_datetime(df["tradingDate"], unit="ms")
                df = df.sort_values("tradingDate")
                if target is None:
                    return float(df.iloc[-1]["close"])
                target_dt = pd.to_datetime(target)
                subset = df[df["tradingDate"] <= target_dt]
                if subset.empty:
                    return None
                return float(subset.iloc[-1]["close"])

            results = []
            for ticker in _normalize_ticker_list(tickers):
                try:
                    start_px = fetch_price(ticker, start_date)
                    end_px = fetch_price(ticker, end_date)
                    if start_px and end_px:
                        performance = ((end_px - start_px) / start_px) * 100
                    else:
                        performance = None
                    results.append(
                        {
                            "ticker": ticker,
                            "start_price": start_px,
                            "end_price": end_px,
                            "performance_pct": performance,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        {
                            "ticker": ticker,
                            "error": str(exc),
                        }
                    )

            return {"results": results, "start_date": start_date, "end_date": end_date}

    def _register_commentary(self) -> None:
        parameters = {
            "ticker": {
                "type": "string",
                "description": "Broker ticker.",
                "required": True,
            },
            "quarter": {
                "type": "string",
                "description": "Quarter label (e.g., 1Q24).",
                "required": True,
            },
            "force_regenerate": {
                "type": "boolean",
                "description": "Ignore cached commentary.",
                "required": False,
            },
        }

        @self._register_tool(
            name="get_commentary",
            description="Generate narrative commentary for a broker quarter.",
            parameters=parameters,
        )
        def _tool(
            ticker: str,
            quarter: str,
            force_regenerate: bool = False,
        ) -> Dict[str, Any]:
            ticker = ticker.upper()

            df = brokerage_data.load_ticker_quarter_data(ticker=ticker, quarter_label=quarter, lookback_quarters=6)
            if df.empty:
                raise ValueError(f"No financial data available for {ticker} {quarter}.")

            commentary = generate_commentary(
                ticker=ticker,
                year_quarter=quarter,
                df=df,
                force_regenerate=force_regenerate,
            )

            return {"ticker": ticker, "quarter": quarter, "commentary": commentary}


def get_tool_specs() -> List[Dict[str, Any]]:
    """Convenience helper to retrieve tool specifications without instantiating twice."""
    return BrokerageMCP().tool_specs
