# MCP (Model Context Protocol) Banking Analysis System

## Overview

The MCP Banking Analysis System integrates OpenAI's GPT models with Vietnamese banking data to provide intelligent, context-aware analysis. The system uses a modular tool architecture that allows OpenAI to access specialized functions and chain them together to answer complex banking questions.

## Current Architecture

```
┌─────────────────────────────────────────────────┐
│        Streamlit UI (pages/7_DucGPT_Chatbot.py) │
│  - Natural language chat interface               │
│  - Tool execution visualization                  │
│  - Conversation memory with compression          │
│  - Results display with formatting               │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         OpenAI Integration Layer                 │
│  - GPT-5 model (configurable via .env)           │
│  - Manages conversation flow                     │
│  - Handles multiple tool calls in parallel       │
│  - Chains tools until answer is complete         │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│      Tool System (utilities/Banking_MCP.py)      │
│  - 8 modular tools with decorator pattern        │
│  - Lazy data loading with caching                │
│  - Universal single/multiple ticker support      │
│  - 5-minute result caching (TTL)                 │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│              Data Layer                          │
│  Primary Data Files:                             │
│  - dfsectorquarter.csv (Historical quarterly)    │
│  - dfsectoryear.csv (Historical yearly)          │
│  - dfsectorforecast.csv (2025-2026 forecasts)   │
│                                                  │
│  Aggregated/Sector Data:                         │
│  - Sector tickers: Sector, SOCB, Private_1/2/3   │
│  - Pre-calculated in same data files             │
│                                                  │
│  Analysis Files:                                 │
│  - earnings_quality_quarterly.csv (QoQ, YoY, T12M)│
│  - earnings_quality_yearly.csv                   │
│  - banking_comments.xlsx (AI commentary)         │
│  - quarterly_analysis_results.xlsx               │
│  - Valuation_banking.csv (52K historical points) │
│                                                  │
│  Reference Files:                                │
│  - Bank_Type.xlsx (Bank classifications)         │
│  - Key_items.xlsx (Metric definitions)           │
└──────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Universal Tool Pattern

Most tools support both single and multiple entity queries:
```python
# Single ticker
get_bank_info(tickers=["VCB"])

# Multiple tickers  
get_bank_info(tickers=["VCB", "ACB", "BID"])

# Response adapts: single returns simple dict, multiple returns batch format
```

### 2. Sector Ticker Support

`query_historical_data` now supports pre-aggregated sector data:
- **"Sector"** - Overall banking sector aggregate
- **"SOCB"** - State-owned commercial banks aggregate
- **"Private_1"** - Tier 1 private banks aggregate
- **"Private_2"** - Tier 2 private banks aggregate
- **"Private_3"** - Tier 3 private banks aggregate

These can be mixed with individual bank tickers in the same query.

### 3. Tool Chaining

OpenAI chains tools automatically to build complete answers:
1. `get_data_availability()` → Understand available periods
2. `query_historical_data()` → Get specific metrics
3. `get_earnings_drivers()` → Analyze profit drivers
4. Continue until analysis is complete (up to 20 tool calls)

### 4. Data Caching Strategy

- **Lazy Loading**: Data files loaded only when needed
- **Memory Cache**: Data kept in memory after first load
- **Result Cache**: Tool results cached for 5 minutes (TTL)
- **Cache Key**: Based on tool name + sorted arguments

## Current Tools (8 Active)

### 1. get_data_availability()
**Purpose**: Discover what data periods are available  
**Parameters**: None  
**Returns**: 
- Current date
- Latest quarterly data (last 8 quarters)
- Latest yearly data (last 5 years)
- Forecast years available
**Usage**: MUST be called first for any "latest", "recent", or "current" data queries

### 2. get_bank_sector_info()
**Purpose**: Get bank/sector information - lists banks, identifies sectors, or returns component banks  
**Parameters**:
- `tickers` (optional): Array of bank tickers OR sector names (SOCB, Private_1, etc.)
**Returns**: 
- No params: All banks grouped by sector
- Bank ticker: Sector classification
- Sector name: Component banks within that sector
**Note**: Merged functionality of list_all_banks and get_bank_info

### 3. query_historical_data()
**Purpose**: Query historical banking metrics with optional filtering  
**Parameters**:
- `frequency` (required): "quarterly" or "yearly"
- `tickers` (optional): Array of bank/sector tickers
- `period` (optional): Single period like "2024-Q3" or "2024"
- `periods` (optional): Multiple periods ["2024-Q1", "2024-Q2"]
- `metric` (optional): Single metric name (efficient for specific queries)
- `metric_group` (optional): "all", "profitability", "asset_quality", "growth"
**Special Features**:
- Supports YTD queries (e.g., "2025-YTD")
- Handles sector tickers (Sector, SOCB, Private_1/2/3)
- Mixed-case ticker handling for compatibility
**Returns**: DataFrame with requested metrics

### 4. query_forecast_data()
**Purpose**: Get forecast data for 2025-2026  
**Parameters**:
- `tickers` (optional): Array of bank tickers
**Returns**: 
- Actual data from latest year
- Forecast data for all future years
- Growth rate comparisons
**Note**: Always returns ALL forecast years (no year filtering)

### 5. get_commentary()
**Purpose**: Get AI-generated analysis and commentary  
**Parameters**:
- `tickers` (required): Array of bank tickers or ["Sector"]
- `quarter` (required): Quarter like "2024-Q3"
**Returns**:
- For banks: Individual AI commentary
- For "Sector": Quarterly market analysis
**Data Source**: banking_comments.xlsx, quarterly_analysis_results.xlsx

### 6. get_valuation_analysis()
**Purpose**: Statistical valuation analysis with Z-scores  
**Parameters**:
- `tickers` (required): Array of bank tickers
- `metric` (optional): "PE" or "PB" (default: "PB")
**Returns**:
- Current value vs historical statistics
- Z-score and percentile rank
- Interpretation (Undervalued/Fair/Overvalued)
**Data Source**: Valuation_banking.csv (52K+ data points)

### 7. get_stock_performance()
**Purpose**: Get stock price performance between dates  
**Parameters**:
- `tickers` (required): Array of stock tickers
- `start_date` (required): YYYY-MM-DD format
- `end_date` (required): YYYY-MM-DD format
**Returns**:
- Start/end prices
- Performance percentage
- Ranking for multiple stocks
**Data Source**: TCBS API (real-time)

### 8. get_earnings_drivers()
**Purpose**: Analyze what's driving profit changes  
**Parameters**:
- `tickers` (required): Array of bank tickers
- `period` (required): Period like "2024-Q3" or "2024"
- `timeframe` (optional): "QoQ", "YoY", "T12M" (quarterly only)
- `frequency` (optional): "quarterly" or "yearly"
**Returns**:
- PBT growth rate
- Revenue, cost, non-recurring impacts
- Detailed component breakdown (NII, fees, OPEX, provisions)
**Data Source**: earnings_quality_quarterly.csv, earnings_quality_yearly.csv

## Removed/Deprecated Tools

### compare_banks (Removed)
**Reason**: Functionality absorbed by query_historical_data with multiple ticker support  
**Migration**: Use `query_historical_data(tickers=["VCB", "ACB", "BID"])`

### get_sector_performance (Removed)
**Reason**: Sector data now accessible via query_historical_data  
**Migration**: Use `query_historical_data(tickers=["Sector"])` or specific sector tickers

### list_all_banks (Removed)
**Reason**: Merged into get_bank_sector_info  
**Migration**: Use `get_bank_sector_info()` with no parameters

### calculate_growth_metrics (Removed)
**Reason**: OpenAI can calculate growth from raw data; pre-calculated growth available in get_earnings_drivers  
**Migration**: 
- Use `query_historical_data()` and let OpenAI calculate growth
- Or use `get_earnings_drivers()` for pre-calculated QoQ, YoY, T12M impacts

## Usage Examples

### Example 1: Simple Query
**User**: "What is VCB's NPL ratio for Q3 2024?"

**System Flow**:
1. `get_data_availability()` → Verify Q3 2024 is available
2. `query_historical_data(frequency="quarterly", tickers=["VCB"], period="2024-Q3", metric="NPL")`
3. Return: NPL ratio of 1.22%

### Example 2: Sector Comparison
**User**: "Compare profitability of state-owned banks vs private banks"

**System Flow**:
1. `query_historical_data(frequency="quarterly", tickers=["SOCB", "Private_1"], period="2024-Q3", metric_group="profitability")`
2. Returns aggregated metrics for both sectors
3. AI analyzes differences in ROA, ROE, NIM

### Example 3: YTD Performance
**User**: "Show me YTD 2025 performance for all banks"

**System Flow**:
1. `get_data_availability()` → Determine completed quarters
2. `query_historical_data(frequency="quarterly", period="2025-YTD", metric_group="all")`
3. Automatically aggregates Q1-Q3 2025 data (if in Q4)

### Example 4: Earnings Analysis
**User**: "What drove VPB's profit growth in Q2 2025?"

**System Flow**:
1. `get_earnings_drivers(tickers=["VPB"], period="2025-Q2", timeframe="QoQ")`
2. Returns structured breakdown:
   - Revenue impact: +15pp (NII: +10pp, Fees: +5pp)
   - Cost impact: -3pp (OPEX: -1pp, Provisions: -2pp)
   - Non-recurring: +2pp

### Example 5: Complex Multi-Tool Analysis
**User**: "Which banks have the best valuation and growth prospects?"

**System Flow**:
1. `get_bank_sector_info()` → Get all bank tickers
2. `get_valuation_analysis(tickers=[...all banks...], metric="PB")` → Parallel execution
3. `query_forecast_data(tickers=[...top 5 undervalued...])` → Growth prospects
4. `get_commentary(tickers=[...top picks...], quarter="2024-Q3")` → Qualitative insights
5. AI synthesizes comprehensive recommendation

### Example 6: Sector Component Query
**User**: "What are the individual banks in the SOCB sector?"

**System Flow**:
1. `get_bank_sector_info(tickers=["SOCB"])`
2. Returns: Component banks ["BID", "CTG", "VCB", "AGB"]

## Adding New Tools

### Tool Definition Pattern
```python
@self.tool(
    name="your_tool_name",
    description="Clear description for OpenAI to understand usage",
    parameters={
        "param1": {
            "type": "string",
            "description": "Parameter description",
            "required": True
        },
        "param2": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional array parameter",
            "required": False
        }
    }
)
def your_tool_name(param1: str, param2: List[str] = None) -> Dict:
    """Implementation"""
    # Use lazy loading for data
    df = self._load_historical_quarter()
    
    # Process request
    result = process_data(df, param1, param2)
    
    # Return structured response
    return {
        "status": "success",
        "data": result,
        "records": len(result)
    }
```

### Best Practices
1. **Lazy Loading**: Load data only when needed
2. **Universal Pattern**: Support single and multiple entities
3. **Structured Returns**: Always return dicts with status
4. **Error Handling**: Return {"error": "message", "status": "failed"}
5. **Efficient Defaults**: Make parameters optional when sensible

## Configuration

### Environment Variables (.env)
```
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-5  # or gpt-4-turbo-preview
```

### Key Settings
- **Model**: GPT-5 (default) or GPT-4 Turbo
- **Temperature**: 1.0 (GPT-5 only supports default)
- **Max Tool Calls**: 20 per conversation turn
- **Cache TTL**: 300 seconds (5 minutes)
- **Conversation Memory**: Last 3 exchanges (compressed)

## Performance Optimizations

### Data Loading
- **Lazy Loading**: Files loaded only when first needed
- **LRU Cache**: @lru_cache decorator on load methods
- **Pre-aggregated Data**: Sector-level data pre-calculated in database

### Tool Execution
- **Parallel Calls**: Multiple banks processed simultaneously
- **Result Caching**: 5-minute TTL on identical queries
- **Compressed Memory**: Conversation history compressed to save tokens

### Pre-calculated Metrics
The database includes pre-calculated growth metrics:
- QoQ (Quarter-over-Quarter)
- YoY (Year-over-Year)
- T12M (Trailing 12 Months)
These should be used instead of recalculating.

## Tool Redundancy Analysis

### Completed Optimizations
1. **Merged list_all_banks into get_bank_sector_info** ✓
2. **Removed calculate_growth_metrics** ✓ (OpenAI calculates from raw data)
3. **Tools reduced from 10 → 8**

### Remaining Opportunities
- Consider combining get_commentary + get_earnings_drivers (both provide analysis)
- Further optimize data loading strategies

## Error Handling

### Common Issues and Solutions

1. **"No data found"**
   - Verify ticker exists in Bank_Type.xlsx
   - Check period format (YYYY-Q# or YYYY)
   - Ensure data files are present

2. **Sector ticker issues**
   - Use exact case: "Sector", not "SECTOR"
   - Private sectors: "Private_1", not "PRIVATE_1"

3. **YTD queries**
   - Format: "2025-YTD"
   - Automatically switches to quarterly frequency
   - Returns all completed quarters in the year

4. **Tool timeout**
   - Large queries may timeout
   - Break into smaller batches
   - Use specific metrics instead of "all"

## Troubleshooting

### Debug Mode
Enable logging by uncommenting debug statements in Banking_MCP.py:
```python
# print(f"DEBUG: Initial data loaded - {len(df)} rows")
```

### Tool Execution Monitoring
The Streamlit UI shows:
- Tool execution order
- Parameters passed
- Success/failure status
- Result summaries
- Execution time

### Cache Inspection
Check session state in Streamlit:
- `st.session_state.tool_cache` - Cached results
- `st.session_state.tool_executions` - Execution history
- `st.session_state.conversation_history` - Compressed chat history

## Future Enhancements

### Immediate Priorities
1. ✓ Tool consolidation complete (10 → 8 tools)
2. Implement streaming responses
3. Add batch processing for large queries

### Medium Term
1. Vietnamese language support
2. Advanced visualizations
3. Export functionality (PDF/Excel)
4. Real-time data integration

### Long Term
1. Machine learning predictions
2. Alert system for significant changes
3. Mobile application
4. Voice interface support