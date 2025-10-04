import streamlit as st
from utils.db import test_connection, get_latest_valuation_data

st.title("🔧 Database Connection Test")

st.info("This page tests the database connection within proper Streamlit context.")

# Test basic connection
st.subheader("Connection Test")

if st.button("Test Database Connection"):
    try:
        with st.spinner("Testing connection..."):
            success = test_connection()

        if success:
            st.success("✅ Database connection successful!")

            # Test a valuation query
            st.subheader("Valuation Data Test")
            try:
                with st.spinner("Testing valuation query..."):
                    # Test with a common Vietnamese stock ticker
                    data = get_latest_valuation_data("VNM")

                if not data.empty:
                    st.success("✅ Valuation query successful!")
                    st.dataframe(data)
                else:
                    st.warning("⚠️ Connection works but no valuation data returned (likely due to firewall)")

            except Exception as query_error:
                st.error(f"❌ Valuation query failed: {query_error}")

        else:
            st.error("❌ Database connection failed")

    except Exception as e:
        st.error(f"❌ Connection test failed: {e}")

# Show current secrets status
st.subheader("Secrets Configuration Status")

try:
    if hasattr(st, 'secrets'):
        if "db" in st.secrets:
            db_config = st.secrets["db"]
            st.success("✅ Database secrets found")

            # Show which keys are available (without showing values)
            available_keys = list(db_config.keys())
            st.info(f"Available configuration keys: {available_keys}")

            # Check for required keys
            required_keys = ['server', 'database', 'username', 'password']
            missing_keys = [key for key in required_keys if key not in db_config]

            if missing_keys:
                st.error(f"❌ Missing required keys: {missing_keys}")
            else:
                st.success("✅ All required database configuration keys present")
        else:
            st.error("❌ No 'db' section found in secrets")
    else:
        st.error("❌ Streamlit secrets not available")

except Exception as e:
    st.error(f"❌ Error checking secrets: {e}")

st.markdown("---")
st.caption("💡 If you see connection errors, check that your secrets.toml file is properly configured and deployed.")