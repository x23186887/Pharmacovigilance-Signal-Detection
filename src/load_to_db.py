import pandas as pd
from sqlalchemy import create_engine

# Connection string — update password if yours is different
DB_URL = "postgresql://postgres:Amritha%402001@localhost:5433/faers_db"
engine = create_engine(DB_URL)

print("Connecting to database...")

# Load each cleaned CSV and push to PostgreSQL
files = {
    "demo": "data/cleaned/demo_clean.csv",
    "drugs": "data/cleaned/drugs_clean.csv",
    "reactions": "data/cleaned/reac_clean.csv",
    "outcomes": "data/cleaned/outc_clean.csv",
    "drug_event_pairs": "data/cleaned/drug_event_pairs.csv"
}

for table_name, filepath in files.items():
    print(f"Loading {table_name}...")
    df = pd.read_csv(filepath, low_memory=False)
    # if_exists='replace' drops and recreates table each run
    df.to_sql(table_name, engine, if_exists='replace', 
              index=False, chunksize=10000)
    print(f" {table_name}: {len(df):,} rows loaded")

print("\nAll tables loaded into PostgreSQL!")