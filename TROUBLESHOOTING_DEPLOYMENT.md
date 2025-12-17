# Troubleshooting Guide: VCI Price Data on Deployed App

## Issue
Prices load successfully on **localhost** but fail to load on **deployed app**.

## Root Cause Analysis
The old TCBS API endpoint (`https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term`) is **no longer available** (returns 404). We've migrated to the **vnstock library** which uses the VCI data source.

## Diagnostic Features Added

### 1. Enhanced Error Messages
- All errors now display in the Streamlit UI (not just console logs)
- Error types and detailed messages are shown
- Errors are tracked in session state

### 2. Diagnostics Panel (in Sidebar)
Click **"🔧 Diagnostics"** in the sidebar to see:

**System Information:**
- ✓ vnstock installation status and version
- Python version
- Pandas version

**Price Cache Status:**
- Number of cached prices
- Last update timestamp

**Error Log:**
- All errors that occurred during price fetching
- Error type and details
- Button to clear error log

**Test API Connection:**
- Button to test VCI connection with a sample ticker (VNM)
- Shows if API is working and returns data

## How to Diagnose on Deployed App

### Step 1: Check Diagnostics Panel
1. Open your deployed app
2. Click the sidebar
3. Click **"🔧 Diagnostics"** expander
4. Check **System Information** section

**Expected:** ✓ vnstock installed: v3.3.0 (or similar)
**If you see:** ✗ vnstock NOT installed → **Go to Solution A**

### Step 2: Test API Connection
1. In the Diagnostics panel
2. Click **"Test VCI Connection"** button
3. Wait for result

**Expected:** ✓ API Connection OK! Fetched X rows
**If you see:** ✗ Connection failed → **Go to Solution B or C**

### Step 3: Check Error Log
1. In the Diagnostics panel
2. Look at **Error Log** section
3. Review the error messages

**Common errors and solutions below**

## Solutions

### Solution A: vnstock Not Installed on Deployed App

**Cause:** Deployment platform didn't install vnstock from requirements.txt

**Fix:**
1. Verify `requirements.txt` contains `vnstock` (it should be line 10)
2. Redeploy your app
3. Check deployment logs for installation errors
4. If using Streamlit Cloud:
   - Go to app settings
   - Click "Reboot app"
   - Watch deployment logs

**For Streamlit Cloud specifically:**
```text
requirements.txt should have:
streamlit
pandas
openpyxl
python-dotenv
pyodbc
openai
plotly
toml
tabulate
vnstock
```

### Solution B: Network/Firewall Restrictions

**Cause:** Deployment platform blocks outbound API calls to VCI

**Symptoms:**
- Error: `Connection timeout`
- Error: `SSLError`
- Error: `Connection refused`

**Fix Options:**

**Option 1:** Check deployment platform's firewall settings
- Streamlit Cloud: Should allow outbound connections (no action needed)
- Heroku: Check outbound firewall rules
- AWS/Azure: Check security groups and network ACLs

**Option 2:** Contact deployment platform support
- Ask if they block connections to Vietnamese APIs
- Request whitelisting for VCI endpoints

**Option 3:** Use alternative data source
- Pre-fetch prices and upload as CSV
- Use a proxy service
- Cache prices in a database

### Solution C: API Rate Limiting

**Cause:** Too many requests to VCI API in short time

**Symptoms:**
- Error: `429 Too Many Requests`
- Error: `Rate limit exceeded`
- Works for first few tickers, then fails

**Fix:**
1. Add delay between requests
2. Reduce number of tickers fetched
3. Use cached prices more aggressively

**Code fix (add to fetch_historical_price function):**
```python
import time
time.sleep(0.5)  # 500ms delay between requests
```

### Solution D: Memory/Timeout Issues

**Cause:** Fetching too many tickers times out on free-tier deployments

**Symptoms:**
- Works for 1-2 tickers
- Times out for full broker list
- Error: `Request timeout`

**Fix:**
1. Increase timeout limits in deployment settings
2. Fetch prices in batches
3. Pre-cache common tickers

### Solution E: Python Version Mismatch

**Cause:** Deployed app uses Python version incompatible with vnstock

**Check:** Diagnostics panel shows Python version

**Fix:**
1. Create `runtime.txt` in project root:
   ```
   python-3.11
   ```
2. Redeploy

### Solution F: Deployment Platform-Specific Issues

#### Streamlit Cloud
- Usually works out of the box
- Check deployment logs for errors
- Reboot app if needed

#### Heroku
- Add to `Procfile`:
  ```
  web: sh setup.sh && streamlit run pages/1_Prop_Book_Dashboard.py
  ```
- Ensure buildpack includes Python

#### AWS/Azure
- Check security groups allow outbound HTTPS
- Verify IAM roles have network permissions

#### Docker
- Ensure Dockerfile installs requirements:
  ```dockerfile
  RUN pip install -r requirements.txt
  ```

## Testing on Localhost vs Production

### Localhost (Working)
- ✓ All Python packages installed
- ✓ No network restrictions
- ✓ Direct internet access

### Production (May Differ)
- ? Package installation depends on requirements.txt
- ? Network access may be restricted
- ? API rate limits may be enforced

## Quick Deployment Checklist

Before deploying, verify:
- [ ] `vnstock` is in `requirements.txt`
- [ ] Python version is 3.7+ (3.11 recommended)
- [ ] `sql/Prop book.xlsx` file is included
- [ ] `.streamlit/config.toml` is included
- [ ] No hardcoded localhost paths

After deploying:
- [ ] Check Diagnostics panel for vnstock installation
- [ ] Test VCI Connection button
- [ ] Click "Refresh Prices" and check Error Log
- [ ] Review deployment logs for errors

## Getting Help

### Information to Provide
When reporting issues, include:
1. **Deployment platform** (Streamlit Cloud, Heroku, AWS, etc.)
2. **Diagnostics panel screenshot** (System Information + Error Log)
3. **Deployment logs** (if accessible)
4. **Error messages** from "Test VCI Connection"

### Where Errors Appear
- **Streamlit UI:** Main error messages with ❌ icons
- **Diagnostics Panel:** Complete error log
- **Deployment Logs:** Installation and system errors
- **Browser Console:** JavaScript errors (usually not relevant)

## Alternative: Pre-fetched Prices

If VCI source continues to fail on deployment, you can pre-fetch prices:

### Option 1: CSV Upload
1. Run script locally to fetch all prices
2. Save to CSV
3. Upload CSV to app
4. Load prices from CSV instead of API

### Option 2: Scheduled Updates
1. Set up scheduled job (cron, GitHub Actions, etc.)
2. Fetch prices and save to database or cloud storage
3. App reads from storage instead of live API

### Option 3: Hybrid Approach
1. Use cached prices by default
2. Only refresh specific tickers on demand
3. Batch update during off-peak hours

---

## Summary

The migration from TCBS API to vnstock/VCI is necessary because the old API no longer exists. The new implementation includes:

✓ Comprehensive error handling
✓ Diagnostic tools in UI
✓ Test connection button
✓ Error logging and tracking

Use the Diagnostics panel to identify the specific issue on your deployment, then apply the appropriate solution above.
