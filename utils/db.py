import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine, text

@st.cache_resource
def get_engine():
    try:
        return create_engine(st.secrets["db"]["url"], pool_pre_ping=True)
    except (KeyError, AttributeError):
        # If secrets not available, raise a more helpful error
        raise RuntimeError("Database connection not configured. Please set up Streamlit secrets in .streamlit/secrets.toml")

def run_query(sql: str, params: dict | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)
