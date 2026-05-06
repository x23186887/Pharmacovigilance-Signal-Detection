# Pharmacovigilance-Signal-Detection

Real-world drug safety analytics pipeline built on FDA FAERS data.

## What This Does
Detects adverse drug reaction signals using disproportionality analysis —
the same methodology used by AstraZeneca, FDA, and EMA safety teams daily.

## Methods
- **ROR** — Reporting Odds Ratio (lower 95% CI > 1)
- **PRR** — Proportional Reporting Ratio (EMA standard)
- **IC** — Information Component (WHO Uppsala Monitoring Centre)
- **ML** — Random Forest + Gradient Boosting signal classifier

## Dataset
- Source: FDA FAERS (free, public)
- Quarters: 25Q1, 25Q2, 25Q3, 25Q4, 26Q1
- Cases: 1.8M deduplicated
- Drug-event pairs: 14.6M
- Drugs analysed: 6 (Vedolizumab, Dupixent, Rituximab, Infliximab, Actemra, Cosentyx)

## Key Results
| Drug | Events | Strong Signals | Max ROR |
|------|--------|---------------|---------|
| VEDOLIZUMAB | 2,741 | 695 | 1,188 |
| DUPIXENT | 2,776 | 471 | — |
| RITUXIMAB | 2,223 | 766 | — |
| INFLIXIMAB | 1,971 | 570 | — |
| ACTEMRA | 1,018 | 322 | — |
| COSENTYX | 1,663 | 518 | 758 |

## ML Performance (External Validation)
| Model | AUC-ROC | F1 |
|-------|---------|-----|
| Gradient Boosting | 0.849 | 0.817 |
| Random Forest | 0.845 | 0.815 |
| Logistic Regression | 0.855 | 0.774 |
| SVM | 0.858 | 0.701 |

## Pipeline
FDA FAERS → ETL → PostgreSQL → ROR/PRR/IC → ML Classifier → Streamlit Dashboard

## Project Structure
src/
load_data.py          # FAERS data ingestion
clean_data.py         # Deduplication + standardisation
load_to_db.py         # PostgreSQL loader
signal_detection.py   # Disproportionality analysis
ml_classifier.py      # Signal classifier
run_more_drugs.py     # Batch signal detection
dashboard/
app.py                # Streamlit dashboard
data/
cleaned/              # Processed CSVs

## MedDRA Terminology
- **PT** — Preferred Term (specific adverse event)
- **SOC** — System Organ Class (body system grouping)
- **HLT** — High Level Term

## Key Pharmacovigilance Concepts
- **Weber Effect** — Reporting spike in first 2 years post-launch
- **Notoriety Bias** — Over-reporting after media coverage
- **Spontaneous Reporting** — Voluntary, no denominator, can't prove causality



