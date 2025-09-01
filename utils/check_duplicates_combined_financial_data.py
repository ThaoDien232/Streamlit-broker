import pandas as pd

def main():
    file_path = 'sql/Combined_Financial_Data.csv'
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Loaded {len(df):,} rows.")

    # Check for fully duplicate rows
    duplicates = df[df.duplicated(keep=False)]
    print(f"Total fully duplicate rows: {len(duplicates)}")
    if not duplicates.empty:
        print("First 10 fully duplicate rows:")
        print(duplicates.head(10).to_string(index=False))
    else:
        print("No fully duplicate rows found.")

if __name__ == "__main__":
    main()
