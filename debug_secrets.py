import streamlit as st

st.title("🔍 Secrets Diagnostic Tool")

st.info("This page helps diagnose secrets configuration issues.")

# Check Streamlit context
st.subheader("1. Streamlit Context Check")
if hasattr(st, 'secrets'):
    st.success("✅ st.secrets is available")

    # Check if secrets exist
    st.subheader("2. Secrets Availability Check")
    try:
        secrets_keys = list(st.secrets.keys())
        st.success(f"✅ Found {len(secrets_keys)} secret sections: {secrets_keys}")

        # Check for db section specifically
        st.subheader("3. Database Configuration Check")
        if "db" in st.secrets:
            st.success("✅ 'db' section found in secrets")

            db_keys = list(st.secrets["db"].keys())
            st.info(f"Database configuration keys: {db_keys}")

            # Check required keys
            required_keys = ['server', 'database', 'username', 'password']
            missing_keys = [key for key in required_keys if key not in db_keys]

            if missing_keys:
                st.error(f"❌ Missing required keys: {missing_keys}")
            else:
                st.success("✅ All required database keys present")

                # Show configuration (without sensitive values)
                st.subheader("4. Configuration Summary")
                config_summary = {}
                for key in db_keys:
                    if key in ['password', 'api_key']:
                        config_summary[key] = "***hidden***"
                    else:
                        config_summary[key] = st.secrets["db"][key]

                st.json(config_summary)

        else:
            st.error("❌ No 'db' section found in secrets")
            st.warning("Available sections:", secrets_keys)

    except Exception as e:
        st.error(f"❌ Error accessing secrets: {e}")

else:
    st.error("❌ st.secrets not available")

# Additional debugging info
st.subheader("5. Environment Info")
import os
st.info(f"Running on: {os.environ.get('STREAMLIT_SHARING_MODE', 'local')}")

st.markdown("---")
st.caption("💡 If you see errors, check your Streamlit Cloud secrets configuration")