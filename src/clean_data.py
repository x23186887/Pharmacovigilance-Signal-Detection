import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_data import load_all

def clean_demo(demo):
    print("Deduplicating cases...")
    demo['fda_dt'] = pd.to_numeric(demo['fda_dt'], errors='coerce')
    demo_clean = (demo
        .sort_values('fda_dt', ascending=False)
        .drop_duplicates(subset='caseid', keep='first')
        .reset_index(drop=True)
    )
    print(f"  Before dedup: {len(demo):,} rows")
    print(f"  After dedup:  {len(demo_clean):,} rows")
    return demo_clean

def clean_drugs(drug, valid_caseids):
    print("Cleaning drug data (memory-efficient)...")
    # Process in chunks to avoid memory error
    chunk_size = 500_000
    ps_chunks = []
    for i in range(0, len(drug), chunk_size):
        chunk = drug.iloc[i:i+chunk_size].copy()
        # Filter to valid cases
        chunk = chunk[chunk['caseid'].isin(valid_caseids)]
        # Filter to Primary Suspect only
        chunk = chunk[chunk['role_cod'] == 'PS']
        # Standardise drug names
        chunk['drugname_clean'] = (chunk['drugname']
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'[^\w\s]', '', regex=True)
        )
        ps_chunks.append(chunk)
        print(f"  Processed {min(i+chunk_size, len(drug)):,} / {len(drug):,} rows...")

    ps_drugs = pd.concat(ps_chunks, ignore_index=True)
    print(f"  Primary Suspect records: {len(ps_drugs):,}")
    return ps_drugs

def clean_reacs(reac, valid_caseids):
    print("Cleaning reaction data...")
    chunk_size = 500_000
    chunks = []
    for i in range(0, len(reac), chunk_size):
        chunk = reac.iloc[i:i+chunk_size].copy()
        chunk = chunk[chunk['caseid'].isin(valid_caseids)]
        chunk['pt_clean'] = chunk['pt'].astype(str).str.upper().str.strip()
        chunks.append(chunk)
    reac_clean = pd.concat(chunks, ignore_index=True)
    print(f"  Reaction records: {len(reac_clean):,}")
    return reac_clean

def clean_outc(outc, valid_caseids):
    print("Cleaning outcomes data...")
    outc_clean = outc[outc['caseid'].isin(valid_caseids)].copy()
    print(f"  Outcome records: {len(outc_clean):,}")
    return outc_clean

def build_drug_event_pairs(ps_drugs, reac):
    print("\nBuilding drug-event pairs table...")
    pairs = pd.merge(
        ps_drugs[['caseid', 'drugname_clean', 'quarter']],
        reac[['caseid', 'pt_clean']],
        on='caseid',
        how='inner'
    )
    pairs.columns = ['caseid', 'drug', 'quarter', 'event']
    pairs = pairs.dropna(subset=['drug', 'event'])
    pairs = pairs[pairs['drug'].str.len() > 0]
    pairs = pairs[pairs['event'].str.len() > 0]
    print(f"  Drug-event pairs: {len(pairs):,}")
    print(f"  Unique drugs: {pairs['drug'].nunique():,}")
    print(f"  Unique events: {pairs['event'].nunique():,}")
    return pairs

if __name__ == "__main__":
    demo, drug, reac, outc = load_all()

    demo_clean = clean_demo(demo)
    valid_ids = set(demo_clean['caseid'])

    # Free memory immediately after use
    del demo
    print("  Freed DEMO from memory")

    ps_drugs = clean_drugs(drug, valid_ids)
    del drug
    print("  Freed DRUG from memory")

    reac_clean = clean_reacs(reac, valid_ids)
    del reac
    print("  Freed REAC from memory")

    outc_clean = clean_outc(outc, valid_ids)
    del outc
    print("  Freed OUTC from memory")

    pairs = build_drug_event_pairs(ps_drugs, reac_clean)

    print("\nSaving cleaned data...")
    os.makedirs("data/cleaned", exist_ok=True)
    demo_clean.to_csv("data/cleaned/demo_clean.csv", index=False)
    ps_drugs.to_csv("data/cleaned/drugs_clean.csv", index=False)
    reac_clean.to_csv("data/cleaned/reac_clean.csv", index=False)
    outc_clean.to_csv("data/cleaned/outc_clean.csv", index=False)
    pairs.to_csv("data/cleaned/drug_event_pairs.csv", index=False)
    print("All cleaned files saved to data/cleaned/ ")
    