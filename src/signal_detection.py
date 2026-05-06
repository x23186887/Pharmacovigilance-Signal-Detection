import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:Amritha%402001@localhost:5433/faers_db"
engine = create_engine(DB_URL)

def get_counts_from_db(drug_name):
    """
    Use SQL to compute contingency table counts directly in the database.
    This avoids loading 14 million rows into Python memory.
    """
    query = text("""
        WITH totals AS (
            SELECT COUNT(*) AS total FROM drug_event_pairs
        ),
        drug_total AS (
            SELECT COUNT(*) AS n_drug 
            FROM drug_event_pairs 
            WHERE drug = :drug
        ),
        event_counts AS (
            SELECT 
                event,
                COUNT(*) AS a,
                (SELECT n_drug FROM drug_total) AS drug_total,
                (SELECT total FROM totals) AS grand_total
            FROM drug_event_pairs
            WHERE drug = :drug
            GROUP BY event
            HAVING COUNT(*) >= 3
        ),
        event_background AS (
            SELECT 
                event,
                COUNT(*) AS event_total
            FROM drug_event_pairs
            WHERE drug != :drug
            GROUP BY event
        )
        SELECT 
            e.event,
            e.a,
            e.drug_total - e.a AS b,
            eb.event_total AS c,
            e.grand_total - e.drug_total - eb.event_total AS d,
            e.grand_total
        FROM event_counts e
        JOIN event_background eb ON e.event = eb.event
    """)
    
    with engine.connect() as conn:
        result = pd.read_sql(query, conn, params={"drug": drug_name})
    return result

def compute_metrics(row):
    """Compute ROR, PRR, IC for one drug-event pair."""
    a, b, c, d = row['a'], row['b'], row['c'], row['d']
    
    # Safety checks
    if any(x <= 0 for x in [a, b, c, d]):
        return pd.Series({
            'ror': np.nan, 'ror_lower_ci': np.nan, 'ror_upper_ci': np.nan,
            'prr': np.nan, 'chi2': np.nan,
            'ic': np.nan, 'ic025': np.nan,
            'signal_ror': False, 'signal_prr': False, 'signal_ic': False,
            'signal_strength': 0
        })
    
    # ROR
    ror = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    ror_lower = np.exp(np.log(ror) - 1.96 * se)
    ror_upper = np.exp(np.log(ror) + 1.96 * se)
    
    # PRR
    prr = (a / (a + b)) / (c / (c + d))
    try:
        chi2 = stats.chi2_contingency([[a, b], [c, d]], correction=False)[0]
    except:
        chi2 = 0
    
    # IC
    n = a + b + c + d
    ic = np.log2((a / n) / ((a + b) / n * (a + c) / n))
    ic025 = ic - 3.3 * np.sqrt(1/a - 1/n)
    
    # Signal flags
    signal_ror = ror_lower > 1
    signal_prr = (prr >= 2) and (chi2 >= 4) and (a >= 3)
    signal_ic  = ic025 > 0
    
    return pd.Series({
        'ror': round(ror, 3),
        'ror_lower_ci': round(ror_lower, 3),
        'ror_upper_ci': round(ror_upper, 3),
        'prr': round(prr, 3),
        'chi2': round(chi2, 3),
        'ic': round(ic, 3),
        'ic025': round(ic025, 3),
        'signal_ror': signal_ror,
        'signal_prr': signal_prr,
        'signal_ic': signal_ic,
        'signal_strength': int(signal_ror) + int(signal_prr) + int(signal_ic)
    })

def run_signal_detection(drug_name):
    print(f"\nRunning signal detection for: {drug_name}")
    print("  Querying database (this may take 1-2 minutes)...")
    
    counts = get_counts_from_db(drug_name)
    print(f"  Found {len(counts)} events with >= 3 cases")
    
    if len(counts) == 0:
        print("  No results found.")
        return pd.DataFrame()
    
    # Compute metrics for each event
    print("  Computing ROR, PRR, IC...")
    metrics = counts.apply(compute_metrics, axis=1)
    results = pd.concat([counts[['event', 'a']], metrics], axis=1)
    results = results.rename(columns={'a': 'n_cases'})
    results['drug'] = drug_name
    
    # Sort by signal strength then ROR
    results = results.sort_values(
        ['signal_strength', 'ror'], ascending=[False, False]
    ).reset_index(drop=True)
    
    strong = (results['signal_strength'] == 3).sum()
    print(f"  Strong signals (all 3 methods): {strong}")
    print(f"  Total drug-event pairs analysed: {len(results)}")
    return results

if __name__ == "__main__":
    # Show top drugs
    print("Top 20 most reported drugs:")
    top_drugs = pd.read_sql("""
        SELECT drug, COUNT(*) as n 
        FROM drug_event_pairs 
        GROUP BY drug 
        ORDER BY n DESC 
        LIMIT 20
    """, engine)
    print(top_drugs.to_string())
    
    # Run for top drug
    target_drug = top_drugs.iloc[0]['drug']
    signals = run_signal_detection(target_drug)
    
    if len(signals) > 0:
        signals.to_csv("data/cleaned/signals_output.csv", index=False)
        print(f"\nSignals saved to data/cleaned/signals_output.csv")
        print("\nTop 10 strongest signals:")
        print(signals.head(10)[['drug','event','n_cases','ror','prr','ic025','signal_strength']].to_string())