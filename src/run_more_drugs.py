import sys, os
sys.path.insert(0, 'src')
os.chdir('D:/projects/healthcare/pharmacovigilance')

from signal_detection import run_signal_detection
import pandas as pd

# Load existing signals
existing = pd.read_csv('data/cleaned/signals_output.csv')
print('Existing drugs:', existing['drug'].unique().tolist())

# Add more drugs here anytime you want
drugs_to_run = ['DUPIXENT', 'RITUXIMAB', 'INFLIXIMAB', 'ACTEMRA', 'COSENTYX']

all_signals = [existing]
for drug in drugs_to_run:
    print(f'\nRunning: {drug}')
    signals = run_signal_detection(drug)
    if len(signals) > 0:
        all_signals.append(signals)
        print(f'  Added {len(signals)} signals for {drug}')
    else:
        print(f'  No signals found for {drug}')

combined = pd.concat(all_signals, ignore_index=True)
combined = combined.drop_duplicates(subset=['drug', 'event'])
combined.to_csv('data/cleaned/signals_output.csv', index=False)

print(f'\nDone!')
print(f'Total drugs in file: {combined["drug"].nunique()}')
print(f'Drugs: {combined["drug"].unique().tolist()}')