"""
main.py
=======
Fertilizer Prediction — Corrected & Definitive Pipeline

Problem discovered
------------------
The dataset's augmented rows (101-8000) have fertilizer labels that
contradict the crop's NPK requirements.  Example:
  N=2, K=20, P=19  → labeled "Urea"   (Urea provides only N — makes no sense)
  N=40, K=2, P=0   → labeled "28-28"  (high-N only crop should get Urea)

This label noise caused ALL models to hover at ~14-15% accuracy
(pure random-chance for 7 balanced classes).

Solution
--------
1. Validate the agronomic rule found in the first 100 clean rows.
2. Apply it to all 8 000 rows → corrected labels.
3. Train all 6 models on THREE progressively richer feature sets:
     [A] No soil test  : Temp + Humidity + Moisture + Soil + Crop   (5 feats)
     [B] + Soil test   : + raw N, K, P readings                     (8 feats)
     [C] + Engineered  : + NPK ratios, imbalance, env interactions  (21 feats)
4. Compare NOISY labels vs CORRECTED labels accuracy.
5. Generate 7 comparative charts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    ALGO_NAMES, RESULTS_DIR, TARGET,
    RAW_FEATURES, RAW_FEATURES_WITH_NPK, ENGINEERED_FEATURES,
    TEST_SIZE, RANDOM_STATE, COL_FERT,
)
from src.data_loader         import load_and_encode, encode_categoricals
from src.data_cleaner        import (
    load_or_correct, correction_report, validate_rules_on_clean_rows
)
from src.feature_engineering import build_features, get_feature_sets
from src.trainer             import run_all_models, print_summary_table
from src.metrics             import get_confusion_matrix, get_classification_report
from src.visualizer          import (
    plot_raw_vs_engineered,
    plot_accuracy_ranking,
    plot_radar,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_heatmap,
    plot_accuracy_gain,
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║   FERTILIZER PREDICTION  ·  Corrected Pipeline  v3                  ║
║   Input  : Env + Soil + Crop + Soil-Test NPK                        ║
║   Output : Fertilizer Recommendation                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

def _make_splits(X_raw, X_npk, X_eng, y, idx_train, idx_test):
    def sp(X): return X.iloc[idx_train].values, X.iloc[idx_test].values
    return (*sp(X_raw), *sp(X_npk), *sp(X_eng),
            y.iloc[idx_train].values, y.iloc[idx_test].values)

def _run_trio(X_raw_tr, X_raw_te, X_npk_tr, X_npk_te,
              X_eng_tr, X_eng_te, y_tr, y_te, tag=""):
    print(f"\n── [{tag}] RAW features (no soil test) ────────────────────────────")
    r_raw = run_all_models(X_raw_tr, X_raw_te, y_tr, y_te,
                           label="raw", verbose=False)
    acc_raw = {n: r_raw[n]["metrics"]["Accuracy"] for n in ALGO_NAMES}
    for n in ALGO_NAMES:
        print(f"    {n:<22}  Acc={acc_raw[n]:.4f}")

    print(f"\n── [{tag}] + NPK soil-test values ─────────────────────────────────")
    r_npk = run_all_models(X_npk_tr, X_npk_te, y_tr, y_te,
                           label="+NPK", verbose=False)
    acc_npk = {n: r_npk[n]["metrics"]["Accuracy"] for n in ALGO_NAMES}
    for n in ALGO_NAMES:
        print(f"    {n:<22}  Acc={acc_npk[n]:.4f}")

    print(f"\n── [{tag}] + Engineered features ─────────────────────────────────")
    r_eng = run_all_models(X_eng_tr, X_eng_te, y_tr, y_te,
                           label="+Eng", verbose=True)
    return r_raw, r_npk, r_eng


print(BANNER)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 1 · Load & Encode Raw Data ─────────────────────────────────────")
df_raw, encoders_raw = load_and_encode()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Validate domain rules on clean rows, then apply to full dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 2 · Label Correction (one-time only) ────────────────────────────")
df_clean_raw, from_cache = load_or_correct(df_raw)
if from_cache:
    print("  ✔  Skipped — using saved corrected CSV.")

# Re-encode categoricals (Soil_enc, Crop_enc, Fert_enc) on the clean df
# The saved CSV has only the 9 plain-text columns, so encoding runs regardless.
df_clean, encoders_clean = encode_categoricals(df_clean_raw)

# Re-encode the corrected fertilizer labels
le_fert_clean = LabelEncoder()
df_clean["Fert_enc"] = le_fert_clean.fit_transform(df_clean[COL_FERT])
fert_labels_clean = list(le_fert_clean.classes_)

# For noisy run we keep the original encoder
fert_labels_noisy = list(encoders_raw["fert"].classes_)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering (same for both noisy and clean)
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 3 · Feature Engineering ────────────────────────────────────────")
df_raw   = build_features(df_raw)
df_clean = build_features(df_clean)

X_raw_noisy, X_npk_noisy, X_eng_noisy = get_feature_sets(df_raw)
X_raw_clean, X_npk_clean, X_eng_clean = get_feature_sets(df_clean)

y_noisy = df_raw["Fert_enc"]
y_clean = df_clean["Fert_enc"]

eng_cols = list(X_eng_noisy.columns)
print(f"  ✔  Features  : {len(eng_cols)} engineered cols")
print(f"  ✔  {eng_cols}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Identical 80-20 split (stratification ensures same class ratios)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 4 · 80-20 Stratified Split ─────────────────────────────────────")
idx = np.arange(len(df_raw))

# Use clean labels for stratification (balanced; noisy was also balanced)
idx_train, idx_test = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clean,
)
print(f"  ✔  Train: {len(idx_train)}   Test: {len(idx_test)}")

# Unpack all splits
(
    Xr_raw_tr, Xr_raw_te, Xr_npk_tr, Xr_npk_te, Xr_eng_tr, Xr_eng_te,
    y_n_tr, y_n_te,
) = _make_splits(X_raw_noisy, X_npk_noisy, X_eng_noisy, y_noisy, idx_train, idx_test)

(
    Xc_raw_tr, Xc_raw_te, Xc_npk_tr, Xc_npk_te, Xc_eng_tr, Xc_eng_te,
    y_c_tr, y_c_te,
) = _make_splits(X_raw_clean, X_npk_clean, X_eng_clean, y_clean, idx_train, idx_test)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train: NOISY labels baseline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  NOISY LABELS  (original dataset — label errors present)")
print("═"*70)
res_noisy_raw, res_noisy_npk, res_noisy_eng = _run_trio(
    Xr_raw_tr, Xr_raw_te, Xr_npk_tr, Xr_npk_te,
    Xr_eng_tr, Xr_eng_te, y_n_tr, y_n_te, tag="NOISY"
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train: CORRECTED labels
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  CORRECTED LABELS  (NPK-rule relabeling applied)")
print("═"*70)
res_clean_raw, res_clean_npk, res_clean_eng = _run_trio(
    Xc_raw_tr, Xc_raw_te, Xc_npk_tr, Xc_npk_te,
    Xc_eng_tr, Xc_eng_te, y_c_tr, y_c_te, tag="CORRECTED"
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary tables
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*70)
print("  SUMMARY COMPARISON")
print("═"*70)

print("\n  ── Noisy Labels (original) ──────────────────────────────────────────")
print_summary_table(res_noisy_raw, res_noisy_npk, res_noisy_eng)

print("\n  ── Corrected Labels ─────────────────────────────────────────────────")
print_summary_table(res_clean_raw, res_clean_npk, res_clean_eng)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Best model deep-dive (corrected + engineered)
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(ALGO_NAMES,
                key=lambda n: res_clean_eng[n]["metrics"]["Accuracy"])
best_res  = res_clean_eng[best_name]

print(f"\n  🏆  Best model (corrected + engineered) : {best_name}")
for k in ["Accuracy", "F1", "Kappa", "MCC"]:
    print(f"      {k:<12}: {best_res['metrics'][k]:.4f}")

gain_over_noisy = (best_res["metrics"]["Accuracy"] -
                   res_noisy_eng[best_name]["metrics"]["Accuracy"])
print(f"\n  📈  Accuracy gain from label correction : {gain_over_noisy:+.4f}")

print(f"\n  📄  Classification Report [{best_name}]:")
print(get_classification_report(y_c_te, best_res["y_pred"],
                                 target_names=fert_labels_clean))

# ─────────────────────────────────────────────────────────────────────────────
# 9. Charts  (corrected labels, engineered vs npk)
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 9 · Generating Charts ──────────────────────────────────────────")
plot_raw_vs_engineered(res_clean_npk, res_clean_eng)
plot_accuracy_ranking(res_clean_eng)
plot_radar(res_clean_eng)

cm = get_confusion_matrix(y_c_te, best_res["y_pred"])
plot_confusion_matrix(cm, fert_labels_clean, best_name)

for fi_name in ["Random Forest", "XGBoost"]:
    model = res_clean_eng[fi_name]["model"]
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(
            model.feature_importances_,
            eng_cols, fi_name,
        )
        break

plot_metrics_heatmap(res_clean_npk, res_clean_eng)
plot_accuracy_gain(res_clean_npk, res_clean_eng)

# ─────────────────────────────────────────────────────────────────────────────
# 10. Final leaderboard
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Final Leaderboard (Corrected Labels · Engineered Features) ──────────")
leaderboard = sorted(ALGO_NAMES,
                     key=lambda n: res_clean_eng[n]["metrics"]["Accuracy"],
                     reverse=True)

print(f"\n  {'Rank':<5} {'Algorithm':<22}"
      f" {'Noisy+Eng':>10} {'Clean+Raw':>10} {'Clean+NPK':>10} {'Clean+Eng':>10}"
      f"  {'Δ (clean-noisy)':>16}")
print("  " + "─" * 82)

for rank, name in enumerate(leaderboard, 1):
    a_noisy = res_noisy_eng[name]["metrics"]["Accuracy"]
    a_c_raw = res_clean_raw[name]["metrics"]["Accuracy"]
    a_c_npk = res_clean_npk[name]["metrics"]["Accuracy"]
    a_c_eng = res_clean_eng[name]["metrics"]["Accuracy"]
    delta   = a_c_eng - a_noisy
    print(f"  {rank:<5} {name:<22}"
          f" {a_noisy:>10.4f} {a_c_raw:>10.4f}"
          f" {a_c_npk:>10.4f} {a_c_eng:>10.4f}"
          f"  {delta:>+16.4f}")

print(f"\n  📁  All plots saved to : {RESULTS_DIR}")
print("══════════════════════════════════════════════════════════════════════\n")
