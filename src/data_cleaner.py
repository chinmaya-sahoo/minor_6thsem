"""
data_cleaner.py
===============
Detects and corrects mislabeled fertilizer entries in the dataset using
agronomic domain rules extracted from the clean rows (first 100 records).

Root cause of label noise
--------------------------
The dataset was created by augmenting ~100 clean records to 8000 rows.
During augmentation, fertilizer labels were NOT re-derived from the
altered N/K/P values — they were either kept from the source row or
randomly re-assigned.  This produces contradictions like:
  N=2, K=20, P=19  →  "Urea"   ← WRONG (Urea provides only N)
  N=40, K=2, P=0   →  "28-28"  ← WRONG (high-N only → Urea)

Domain rule (N, K, P represent crop nutrient REQUIREMENTS)
-----------------------------------------------------------
Choose the fertilizer whose macro-nutrient composition best matches
the crop's requirement profile:

  Urea     (46-0-0)  : N needed heavily, K≈0, P≈0
  DAP      (18-46-0) : P dominant, some N, K≈0
  14-35-14           : P dominant WITH meaningful K
  17-17-17           : balanced N ≈ K ≈ P
  28-28    (28-0-28) : moderate N + P, K≈0, N ≥ P
  20-20    (20-0-20) : low N + P, K≈0
  10-26-26           : K ≈ P dominant, both dominate N
"""

import numpy as np
import pandas as pd

from src.config import COL_N, COL_K, COL_P, COL_FERT, CLEAN_DATA_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Core rule
# ─────────────────────────────────────────────────────────────────────────────

def _assign_fertilizer(N: float, K: float, P: float) -> str:
    """
    Deterministically assign fertilizer based on the NPK requirement profile.

    Parameters
    ----------
    N : Nitrogen requirement (higher → more N needed)
    K : Potassium requirement
    P : Phosphorous requirement

    Returns
    -------
    Fertilizer name string (one of the 7 classes in the dataset)
    """
    # ── Branch 1 : No meaningful potassium requirement ─────────────────────
    if K < 3:
        if N >= 30 and P < 5:
            return "Urea"          # Only N needed (e.g. N=37, K=0, P=0)
        elif P >= 30:
            return "DAP"           # High phosphorous needed  (e.g. N=12, P=36)
        elif N >= 18 and P >= 14 and N >= P:
            return "28-28"         # Moderate N+P, N ≥ P    (e.g. N=22, P=20)
        else:
            return "20-20"         # Low N+P, no K           (e.g. N=9, P=10)

    # ── Branch 2 : Potassium is required ──────────────────────────────────
    balanced      = (abs(N - K) <= 7 and abs(K - P) <= 7 and abs(N - P) <= 7)
    k_p_close     = abs(K - P) < max(K, 1) * 0.45   # K ≈ P within 45%
    both_dom_n    = K > N * 1.4 and P > N * 1.4     # K and P both >> N
    p_dom_k       = P > K * 1.5 and P > N            # P clearly dominates K

    if balanced and not (k_p_close and both_dom_n):
        return "17-17-17"          # Balanced  (e.g. N=12, K=10, P=13)
    elif k_p_close and both_dom_n:
        return "10-26-26"          # K ≈ P, both dominate N (e.g. N=5, K=18, P=15)
    elif p_dom_k:
        return "14-35-14"          # P dominates K  (e.g. N=7, K=9, P=30)
    else:
        return "17-17-17"          # Default balanced


# ─────────────────────────────────────────────────────────────────────────────
# Batch application
# ─────────────────────────────────────────────────────────────────────────────

def correct_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply NPK domain rules to every row and return a DataFrame with a
    corrected 'Fertilizer Name' column.

    Also adds a 'Label_Changed' boolean column for diagnostic purposes.
    Original labels are preserved in 'Fertilizer_Original'.
    """
    df = df.copy()
    df["Fertilizer_Original"] = df[COL_FERT]

    corrected = df.apply(
        lambda row: _assign_fertilizer(row[COL_N], row[COL_K], row[COL_P]),
        axis=1,
    )

    df["Label_Changed"] = corrected != df[COL_FERT]
    df[COL_FERT]        = corrected
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def correction_report(df_cleaned: pd.DataFrame) -> None:
    """Print a brief label-correction diagnostic to stdout."""
    n_changed = df_cleaned["Label_Changed"].sum()
    n_total   = len(df_cleaned)
    pct       = 100 * n_changed / n_total

    print(f"\n  ──────────────────────────────────────────────")
    print(f"  Label Correction Report")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Total rows           : {n_total}")
    print(f"  Labels corrected     : {n_changed}  ({pct:.1f}%)")
    print(f"  Labels unchanged     : {n_total - n_changed}  ({100-pct:.1f}%)")

    # Show old → new distribution
    changed = df_cleaned[df_cleaned["Label_Changed"]]
    if len(changed):
        cross = pd.crosstab(
            changed["Fertilizer_Original"],
            changed["Fertilizer Name"],
            rownames=["Original"],
            colnames=["Corrected"],
        )
        print(f"\n  Correction cross-tab (rows changed only):\n")
        print(cross.to_string(line_width=80))

    print()
    print(f"  Corrected label distribution:")
    print(df_cleaned["Fertilizer Name"].value_counts().to_string())
    print(f"  ──────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Validation: test rules against the 100 known-clean rows
# ─────────────────────────────────────────────────────────────────────────────

def validate_rules_on_clean_rows(df_original: pd.DataFrame) -> float:
    """
    Apply rules to first 100 rows (which are clean) and report match rate.
    Returns the fraction correct (should be ≥ 0.95 for the rule to be trusted).
    """
    clean = df_original.head(100).copy()
    predicted = clean.apply(
        lambda r: _assign_fertilizer(r[COL_N], r[COL_K], r[COL_P]),
        axis=1,
    )
    match_rate = (predicted == clean[COL_FERT]).mean()
    print(f"  ✔  Rule validation on first 100 (clean) rows: "
          f"{match_rate*100:.1f}% match")
    return match_rate


# ─────────────────────────────────────────────────────────────────────────────
# Persist & smart-load  (run correction ONCE, reuse CSV after that)
# ─────────────────────────────────────────────────────────────────────────────

def save_clean_csv(df_cleaned: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    """
    Save the corrected DataFrame to disk.

    Columns saved: only the original 9 CSV columns (diagnostic columns like
    'Label_Changed' and 'Fertilizer_Original' are dropped so the file is
    identical in structure to data_core.csv).
    """
    keep_cols = [
        "Temparature", "Humidity", "Moisture",
        "Soil Type", "Crop Type",
        "Nitrogen", "Potassium", "Phosphorous",
        "Fertilizer Name",
    ]
    df_cleaned[keep_cols].to_csv(path, index=False)
    print(f"  ✔  Corrected dataset saved → {path}")


def load_or_correct(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Smart loader: returns the corrected DataFrame.

    Behaviour
    ---------
    - If  data/data_core_cleaned.csv  already exists → load it directly.
      (skips computation entirely)
    - If it does NOT exist → run correction + save → return corrected df.

    Returns
    -------
    (df_clean, from_cache)
      df_clean   : corrected DataFrame (same structure as df_raw)
      from_cache : True if loaded from file, False if correction was run
    """
    import os
    if os.path.exists(CLEAN_DATA_PATH):
        df_clean = pd.read_csv(CLEAN_DATA_PATH)
        print(f"  ✔  Loaded cached clean data → {CLEAN_DATA_PATH}")
        print(f"     ({len(df_clean)} rows, skipped re-correction)")
        return df_clean, True

    # First time: run correction, save, return
    print("  ⚙  Clean CSV not found — running label correction (one-time only)…")
    match_rate = validate_rules_on_clean_rows(df_raw)
    if match_rate < 0.90:
        print("  ⚠  Rule match < 90% on clean rows — check rule thresholds.")

    df_clean = correct_labels(df_raw)
    correction_report(df_clean)
    save_clean_csv(df_clean)
    return df_clean, False
