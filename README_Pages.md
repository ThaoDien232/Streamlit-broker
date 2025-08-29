# Multi-Page App Structure

This project uses Streamlit's built-in multi-page functionality. Here's how to use and extend it:

## Current Structure

```
Broker-page/
├── streamlit_app.py                 # Main entry point (Home page)
└── pages/                          # All additional pages go here
    ├── 1_Prop_Book_Dashboard.py    # Proprietary trading analysis
    ├── 2_Charts.py                 # Financial charts and visualization
    └── 3_Forecast.py               # Financial forecasting models
```

## Running the App

```bash
streamlit run streamlit_app.py
```

This will automatically discover all pages in the `pages/` folder and create navigation in the sidebar.

## Adding New Pages

To add a new page:

1. **Create a new .py file** in the `pages/` folder
2. **Use numbered prefixes** to control the order (e.g., `4_New_Page.py`)
3. **Include proper page config** at the top of your file:

```python
import streamlit as st

st.set_page_config(page_title="Your Page Name", layout="wide")

st.title("Your Page Title")
# Your page content here...
```

## Page Naming Convention

- **Filename**: `{number}_{Page_Name}.py`
- **Examples**: 
  - `4_Risk_Analysis.py` → "Risk Analysis" in sidebar
  - `5_Market_Data.py` → "Market Data" in sidebar
  - `6_Portfolio_Management.py` → "Portfolio Management" in sidebar

## Page Requirements

Each page should:
- Include `st.set_page_config()` at the very top
- Have a clear title using `st.title()`
- Handle data loading with `@st.cache_data` for performance
- Include error handling for missing data files

## Current Pages Description

- **Home** (`streamlit_app.py`): Welcome page and navigation overview
- **Prop Book Dashboard**: Analyze broker proprietary trading positions with real-time P&L
- **Charts**: Interactive financial charts with quarterly/annual views
- **Forecast**: Financial modeling and forecasting tools

## Tips for Development

1. **Data Loading**: Use `@st.cache_data` decorator for expensive operations
2. **File Paths**: Use relative paths from the project root
3. **Error Handling**: Always include try/catch blocks for file operations
4. **Performance**: Cache data loading functions to improve user experience
5. **UI Consistency**: Use similar styling and layout across pages