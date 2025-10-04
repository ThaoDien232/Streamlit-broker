# 🚀 Deployment Guide

This guide explains how to securely deploy your Streamlit app with database connectivity.

## 🔒 Security Approach

Your app uses a **secure multi-layer configuration strategy**:

1. **Streamlit Cloud Secrets** (recommended for Streamlit Cloud)
2. **Environment Variables** (fallback for other platforms)
3. **Local secrets.toml** (development only - never committed to git)

## 📋 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Deploy your app to Streamlit Cloud**
2. **Configure secrets in the dashboard**:
   - Go to your app dashboard
   - Click **Settings** → **Secrets**
   - Add the following configuration:

```toml
[db]
server = "dcdwhprod.database.windows.net"
database = "dclab"
username = "dclab_readonly"
password = "DHS#@vGESADdf!"
driver = "ODBC Driver 18 for SQL Server"
port = 1433
url = "mssql+pyodbc://dclab_readonly:DHS%23%40vGESADdf%21@dcdwhprod.database.windows.net:1433/dclab?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30"

[openai]
api_key = "your-openai-api-key-here"
```

3. **Save and redeploy** your app

### Option 2: Other Platforms (Heroku, Railway, etc.)

Set these environment variables in your deployment platform:

```bash
DB_SERVER=dcdwhprod.database.windows.net
DB_DATABASE=dclab
DB_USERNAME=dclab_readonly
DB_PASSWORD=DHS#@vGESADdf!
DB_PORT=1433
DB_URL=mssql+pyodbc://dclab_readonly:DHS%23%40vGESADdf%21@dcdwhprod.database.windows.net:1433/dclab?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no&Connection+Timeout=30
OPENAI_API_KEY=your-openai-api-key-here
```

## 🔧 Configuration Priority

Your app will use configuration in this order:

1. **Streamlit Secrets** (`st.secrets["db"]`) - **Preferred**
2. **Environment Variables** (`DB_SERVER`, `DB_DATABASE`, etc.) - **Fallback**
3. **Local secrets.toml** - **Development only**

## ✅ Verification

After deployment, your app will show clear error messages if configuration is missing:

- ✅ **"Database connection successful!"** - Everything working
- ❌ **"Database configuration missing from Streamlit secrets"** - Add secrets to dashboard
- ❌ **"Missing required database configuration: [...]"** - Check which values are missing

## 🛡️ Security Best Practices

### ✅ What We Do (Secure)
- Keep `secrets.toml` in `.gitignore`
- Use Streamlit Cloud secrets dashboard
- Use environment variables for other platforms
- Never commit credentials to git

### ❌ What We Don't Do (Insecure)
- Commit `secrets.toml` to git repository
- Put credentials directly in code
- Share credentials in plain text
- Use same credentials for dev/prod

## 🚨 Troubleshooting

### Database Connection Issues

1. **"st.secrets has no key 'db'"**
   - Add database configuration to Streamlit Cloud secrets
   - Or set environment variables if not using Streamlit Cloud

2. **"All database connection methods failed"**
   - Check Azure SQL firewall allows your deployment IP
   - Verify credentials are correct
   - Check network connectivity

3. **"Missing required database configuration"**
   - Ensure all required fields are set: server, database, username, password

### Testing Your Setup

Run the database test page:
```bash
streamlit run test_streamlit_db.py
```

This will show you exactly what configuration is available and test the connection.

## 📞 Support

If you encounter issues:
1. Check the error messages in your app
2. Verify your secrets are configured correctly
3. Test database connectivity from your deployment platform
4. Check Azure SQL firewall settings