## Brokerage MCP Integration Summary

### Overview
- `utils/brokerage_mcp.BrokerageMCP` remains the tool registry with five-minute server-side caching and JSON-friendly outputs.
- `pages/8_Brokerage_Chatbot.py` now streams model responses, batches tool call round-trips, and records execution history with richer context.
- The chat surface includes context compression, usage tracking, and optional developer-mode reloads to speed MCP iterations.

### Available Tools
- `get_data_availability`: Returns latest actual quarter, annual coverage, and forecast years from `BrokerageMetrics`.
- `get_broker_info`: Lists supported tickers or details availability ranges for requested brokers.
- `query_historical_data`: Pulls quarterly or annual metrics with optional period/metric filters and group shortcuts.
- `query_forecast_data`: Serves forecast metrics (ACTUAL=0) for requested brokers and key income lines.
- `get_valuation_analysis`: Computes percentile-based valuation statistics using the SQL-backed valuation dataset.
- `get_stock_performance`: Fetches price performance between two dates via the TCBS public API.
- `get_commentary`: Generates OpenAI commentary for a broker-quarter using existing financial loaders.

### Key Behaviors
- Tool outputs are cached for 5 minutes server-side and re-used client-side with an additional UI cache governed by `UI_TOOL_CACHE_TTL`.
- Tool specs feed directly into the OpenAI `tools` schema, enabling automatic multi-tool reasoning with streaming completion support.
- Tool execution logs capture cache provenance, row counts, and timestamps, and can be exported from the sidebar.
- Conversation state is compressed to maintain short context windows while still surfacing relevant ticket and period summaries.
- Model selection reads from `OPENAI_MODEL` or Streamlit secrets, defaulting to `gpt-5-mini`, and accumulates token usage plus estimated spend per session.

### Next Steps & Considerations
- Add sector-tier aggregations once supporting tables are available.
- Extend `get_commentary` with richer context (market share, prop book) when data sources are exposed.
- Introduce throttling or pagination for large historical queries if user feedback indicates latency issues.
- Consider a dedicated valuation trend tool when broader metrics (e.g., EV/EBITDA) become consistent across brokers.
- Evaluate lightweight chart-rendering tools to pair with the new streaming surface when chart-ready data lands.
