
import pandas as pd

# Path to your CSV file
csv_path = 'sql/Combined_Financial_Data.csv'

# Load the CSV file
df = pd.read_csv(csv_path, low_memory=False)

# Ensure YEARREPORT and LENGTHREPORT are always integer (drop rows where conversion fails)
df['YEARREPORT'] = pd.to_numeric(df['YEARREPORT'], errors='coerce').astype('Int64')
df['LENGTHREPORT'] = pd.to_numeric(df['LENGTHREPORT'], errors='coerce').astype('Int64')
df = df.dropna(subset=['YEARREPORT', 'LENGTHREPORT'])
df['YEARREPORT'] = df['YEARREPORT'].astype(int)
df['LENGTHREPORT'] = df['LENGTHREPORT'].astype(int)

# Remove all years before 2016
df = df[df['YEARREPORT'] >= 2016]

# Remove specified tickers
tickers_to_remove = ['ABSC', 'APSC', 'ASIAS', 'ATSC', 'AVS','BMSC','CBVS','CLS','CVS','CBVS','DDSC','DNSC','DNSE','DTDS','ECCS','EPS','FLCS','GBS','GLS','GLSC','HASC','HBBS','HBSC','HPC','HSSC','HRSC','HVS','HVSC','IRSC','ISC','JSI','JISC','JSIC','KLS','KEVS','LVSC','MHBS','MKSC','MSBS','MSGS','NASC','NAVS','NHSV','NSIC','NVSC','OCSC','PBSV','PCSC','PGSC','ROSE','RUBSE','SJCS','SME','STSC','SVS','TAS','TCSC','TFSC','TLSC','TSSC','VCSC','VDSE','VFSC','VGSC','VIETS','VNIS','VNSC','VPBS','VQSC','VSEC','VSM','VSMC','VTSC']
df = df[~df['TICKER'].isin(tickers_to_remove)]

# Save back to the same file (overwrite)
df.to_csv(csv_path, index=False)

print('Cleaned Combined_Financial_Data.csv successfully.')
