import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os

# ── Step 1: Load signals ──────────────────────────────────────────────────────
print("Loading signal data...")
signals_df = pd.read_csv("data/cleaned/signals_output.csv")
print(f"Total drug-event pairs: {len(signals_df)}")

# ── Step 2: External validation labels ───────────────────────────────────────
# These are REAL known signals/non-signals for Vedolizumab
# Source: FDA drug safety communications + published pharmacovigilance literature

known_true_signals = [
    "PROGRESSIVE MULTIFOCAL LEUKOENCEPHALOPATHY",
    "LIVER INJURY",
    "INFUSION RELATED REACTION",
    "SERIOUS INFECTION",
    "COLORECTAL CANCER STAGE III",
    "HEPATITIS B VIRUS REACTIVATION",
    "ANAPHYLACTIC REACTION",
    "SEPSIS",
    "OPPORTUNISTIC INFECTION",
    "PNEUMONIA",
    "COLITIS",
    "INTESTINAL OBSTRUCTION",
    "ARTHRALGIA",
    "NASOPHARYNGITIS",
]

known_false_signals = [
    "HEADACHE",
    "NAUSEA",
    "FATIGUE",
    "DIZZINESS",
    "PAIN",
    "ANXIETY",
    "INSOMNIA",
    "BACK PAIN",
    "COUGH",
    "DIARRHOEA",
]

def assign_external_label(event):
    if event in known_true_signals:
        return 1
    elif event in known_false_signals:
        return 0
    else:
        return np.nan  # unknown — exclude from external validation

signals_df['external_label'] = signals_df['event'].apply(assign_external_label)

# How many did we label externally?
labelled = signals_df.dropna(subset=['external_label'])
print(f"\nExternally labelled pairs: {len(labelled)}")
print(f"  True signals:  {int(labelled['external_label'].sum())}")
print(f"  False signals: {int((labelled['external_label']==0).sum())}")

# ── Step 3: Feature matrix ────────────────────────────────────────────────────
FEATURE_COLS = ['n_cases', 'ror', 'ror_lower_ci', 'ror_upper_ci',
                'prr', 'chi2', 'ic', 'ic025', 'signal_strength']

def build_features(df):
    X = df[FEATURE_COLS].copy().fillna(0)
    return X

# ── Step 4: Two label strategies ─────────────────────────────────────────────

# Strategy A — Rule-based labels (from disproportionality metrics)
# Note: this causes label leakage but is useful as a baseline
def rule_based_labels(df):
    return ((df['signal_strength'] == 3) & (df['n_cases'] >= 5)).astype(int)

# Strategy B — External labels (ground truth from published safety alerts)
def external_labels(df):
    return df['external_label']

# ── Step 5: Model comparison function ────────────────────────────────────────
def compare_models(X, y, label_type="Rule-based"):
    print(f"\n{'='*60}")
    print(f"Model Comparison — Labels: {label_type}")
    print(f"{'='*60}")
    print(f"Dataset size: {len(X)} | Positives: {y.sum()} ({y.mean()*100:.1f}%)")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=42),
        "SVM": SVC(
            kernel='rbf', class_weight='balanced', 
            probability=True, random_state=42),
    }

    # Stratified K-Fold — better than basic KFold for imbalanced data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['roc_auc', 'f1', 'precision', 'recall']

    results = []
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        # Scale features (important for LR and SVM)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring)
        
        results.append({
            "Model":     name,
            "AUC-ROC":   round(scores['test_roc_auc'].mean(), 3),
            "AUC-Std":   round(scores['test_roc_auc'].std(), 3),
            "F1":        round(scores['test_f1'].mean(), 3),
            "Precision": round(scores['test_precision'].mean(), 3),
            "Recall":    round(scores['test_recall'].mean(), 3),
        })
        print(f"    AUC-ROC: {results[-1]['AUC-ROC']:.3f} ± {results[-1]['AUC-Std']:.3f}")
        print(f"    F1:      {results[-1]['F1']:.3f}")

    comparison_df = pd.DataFrame(results).sort_values("AUC-ROC", ascending=False)
    print(f"\n{'─'*60}")
    print("FINAL COMPARISON TABLE:")
    print(comparison_df.to_string(index=False))
    return comparison_df

# ── Step 6: Train best model and save ────────────────────────────────────────
def train_best_model(X, y):
    print("\nTraining final Random Forest on full dataset...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    print("\nTest Set Performance (Random Forest):")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Noise','Signal']))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))
    
    # Save model and scaler
    with open("src/signal_classifier.pkl", "wb") as f:
        pickle.dump({'model': clf, 'scaler': scaler}, f)
    print("\nModel saved to src/signal_classifier.pkl ✅")
    
    return clf, scaler

# ── Step 7: Score all signals ─────────────────────────────────────────────────
def score_all_signals(signals_df, clf, scaler):
    X = build_features(signals_df)
    X_scaled = scaler.transform(X)
    signals_df = signals_df.copy()
    signals_df['ml_score'] = clf.predict_proba(X_scaled)[:, 1]
    signals_df['ml_signal'] = clf.predict(X_scaled)
    return signals_df.sort_values('ml_score', ascending=False)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # --- EXPERIMENT A: Rule-based labels (baseline) ---
    X_all = build_features(signals_df)
    y_rule = rule_based_labels(signals_df)
    comparison_rule = compare_models(X_all, y_rule, label_type="Rule-based (leakage warning)")
    comparison_rule.to_csv("data/cleaned/model_comparison_rulebased.csv", index=False)

    # --- EXPERIMENT B: External labels (ground truth) ---
    # Only use rows we've externally labelled
    labelled_df = signals_df.dropna(subset=['external_label']).copy()
    
    if len(labelled_df) >= 10:
        X_ext = build_features(labelled_df)
        y_ext = labelled_df['external_label'].astype(int)
        comparison_ext = compare_models(X_ext, y_ext, label_type="External (ground truth)")
        comparison_ext.to_csv("data/cleaned/model_comparison_external.csv", index=False)
    else:
        print("\nNot enough externally labelled rows for Experiment B.")
        print("Add more known signals/non-signals to the lists above.")

    # --- Train and save best model ---
    clf, scaler = train_best_model(X_all, y_rule)

    # --- Score all signals ---
    scored = score_all_signals(signals_df, clf, scaler)
    scored.to_csv("data/cleaned/signals_scored.csv", index=False)

    print("\n" + "="*60)
    print("TOP 10 ML-RANKED SIGNALS:")
    print("="*60)
    print(scored.head(10)[['drug','event','n_cases','ror',
                            'signal_strength','ml_score']].to_string())
    print("\nAll done! ✅")
    print("Files saved:")
    print("  data/cleaned/signals_scored.csv")
    print("  data/cleaned/model_comparison_rulebased.csv")
    print("  data/cleaned/model_comparison_external.csv")