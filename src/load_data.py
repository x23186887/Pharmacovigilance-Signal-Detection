import pandas as pd
import os

# Define which quarters to load
QUARTERS = {
    "25Q1": 'D:/projects/healthcare/pharmacovigilance/data/raw/faers_25Q1/ASCII',
    "25Q2": 'D:/projects/healthcare/pharmacovigilance/data/raw/faers_25Q2/ASCII',
    "25Q3": 'D:/projects/healthcare/pharmacovigilance/data/raw/faers_25Q3/ASCII',
    "25Q4": 'D:/projects/healthcare/pharmacovigilance/data/raw/faers_25Q4/ASCII',
    "26Q1": 'D:/projects/healthcare/pharmacovigilance/data/raw/faers_26Q1/ASCII'
}

def load_faers_file(file_prefix, sep="$"):
    frames = []
    for quarter, folder in QUARTERS.items():
        filepath = os.path.join(folder, f"{file_prefix}{quarter}.txt")
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping.")
            continue
        print(f"  Loading {filepath}...")
        df = pd.read_csv(filepath, sep=sep, encoding="latin-1",
                         low_memory=False, on_bad_lines='skip')
        df['quarter'] = quarter
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        raise FileNotFoundError("No files found. Check your folder names.")

def load_all():
    print("Loading DEMO (demographics)...")
    demo = load_faers_file("DEMO")
    print("Loading DRUG...")
    drug = load_faers_file("DRUG")
    print("Loading REAC (reactions)...")
    reac = load_faers_file("REAC")
    print("Loading OUTC (outcomes)...")
    outc = load_faers_file("OUTC")
    print("\nShapes loaded:")
    print(f"  DEMO: {demo.shape}")
    print(f"  DRUG: {drug.shape}")
    print(f"  REAC: {reac.shape}")
    print(f"  OUTC: {outc.shape}")
    return demo, drug, reac, outc

if __name__ == "__main__":
    demo, drug, reac, outc = load_all()
    print("\nSample DRUG columns:", drug.columns.tolist())
    print("Sample REAC columns:", reac.columns.tolist())