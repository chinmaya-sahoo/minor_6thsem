"""
feature_engineering.py
=======================
All domain-knowledge-driven feature transformations live here.
Each transform is a pure function:  (DataFrame) → DataFrame

Why NPK is now an INPUT
-----------------------
Fertilizer selection in agriculture is always made AFTER a soil test.
The soil test measures current Nitrogen (N), Potassium (K), and
Phosphorous (P) levels in the field.  The lab result drives the
fertilizer choice — e.g. if N is already high, you don't add Urea.
So N, K, P are correctly treated as inputs (soil-test readings),
not as targets.

Features added
--------------
From NPK readings:
  NPK_total      – total macro-nutrient load in the field
  NPK_N_ratio    – fraction of total that is nitrogen
  NPK_K_ratio    – fraction of total that is potassium
  NPK_P_ratio    – fraction of total that is phosphorous
  NPK_imbalance  – max / (min + 1) → how skewed the profile is
  NPK_dominant   – which nutrient dominates (0=N, 1=K, 2=P)

From environment:
  Temp_bin           – Growing-season category from temperature
  Temp_x_Moisture    – Heat × water availability interaction
  Humidity_x_Soil    – Moisture retention × soil type
  Temp_x_Humidity    – Climatic stress index
  Crop_Soil_interact – Crop-Soil compatibility (joint encoding)
  Heat_Stress_idx    – Temperature / (Humidity + 1)
  Moisture_Soil_idx  – Water-retention capacity by soil class
"""

import numpy as np
import pandas as pd

from src.config import (
    COL_TEMP, COL_HUMIDITY, COL_MOISTURE,
    COL_N, COL_K, COL_P,
    RAW_FEATURES, RAW_FEATURES_WITH_NPK, ENGINEERED_FEATURES,
)


# ─────────────────────────────────────────────────────────────────────────────
# NPK ratio & imbalance features
# ─────────────────────────────────────────────────────────────────────────────

def add_npk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive interpretable ratios and balance metrics from raw N, K, P readings.

    NPK_total     : Sum of available macro-nutrients → overall soil richness.
    NPK_N/K/P_ratio: Share of each nutrient in total → fertilizer type hint.
                   (e.g. all-N profile → Urea is recommended)
    NPK_imbalance : max / (min + 1) → very high value means one nutrient
                   dominates; models can learn that extreme imbalance maps
                   to single-nutrient fertilizers (Urea, DAP).
    NPK_dominant  : Integer label for the highest nutrient (0=N, 1=K, 2=P)
                   → discrete signal directly encoding fertilizer type.
    """
    df = df.copy()
    N  = df[COL_N].astype(float)
    K  = df[COL_K].astype(float)
    P  = df[COL_P].astype(float)

    total = N + K + P + 1e-6        # avoid divide-by-zero

    df["NPK_total"]     = N + K + P
    df["NPK_N_ratio"]   = N / total
    df["NPK_K_ratio"]   = K / total
    df["NPK_P_ratio"]   = P / total
    df["NPK_imbalance"] = df[[COL_N, COL_K, COL_P]].max(axis=1) / \
                          (df[[COL_N, COL_K, COL_P]].min(axis=1) + 1)
    df["NPK_dominant"]  = df[[COL_N, COL_K, COL_P]].values.argmax(axis=1)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Temperature bins  (Growing Season Categories)
# ─────────────────────────────────────────────────────────────────────────────

_TEMP_BINS   = [0,   25,  31,  36, 100]
_TEMP_LABELS = [0,    1,   2,   3]       # 0=cold, 1=moderate, 2=warm, 3=hot


def add_temp_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin temperature into four growing-season categories.
      cold     : < 25 °C
      moderate : 25–31 °C
      warm     : 31–36 °C
      hot      : ≥ 36 °C
    """
    df = df.copy()
    df["Temp_bin"] = pd.cut(
        df[COL_TEMP],
        bins=_TEMP_BINS,
        labels=_TEMP_LABELS,
        right=False,
    ).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Environment interaction terms
# ─────────────────────────────────────────────────────────────────────────────

def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplicative interactions capturing joint effects of env variables.
    """
    df = df.copy()
    df["Temp_x_Moisture"] = df[COL_TEMP]     * df[COL_MOISTURE]
    df["Humidity_x_Soil"] = df[COL_HUMIDITY] * df["Soil_enc"]
    df["Temp_x_Humidity"] = df[COL_TEMP]     * df[COL_HUMIDITY]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Crop-Soil compatibility score
# ─────────────────────────────────────────────────────────────────────────────

def add_crop_soil_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the joint (Crop, Soil) pair as a single integer.
    Different pairs have distinct fertilizer requirements regardless of NPK.
    """
    df = df.copy()
    n_soil = df["Soil_enc"].max() + 1
    df["Crop_Soil_interact"] = df["Crop_enc"] * n_soil + df["Soil_enc"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stress & efficiency indices  (no longer using NPK as proxy — real values used)
# ─────────────────────────────────────────────────────────────────────────────

def add_stress_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Environmental stress proxies.

    Heat_Stress_idx  = Temp / (Humidity + 1)
        High temp + low humidity → crop stress → affects nutrient uptake.
    Moisture_Soil_idx = Moisture / (Soil_enc + 1)
        Water-retention capacity: clayey soils (high enc) hold more moisture.
    """
    df = df.copy()
    df["Heat_Stress_idx"]   = df[COL_TEMP] / (df[COL_HUMIDITY] + 1)
    df["Moisture_Soil_idx"] = df[COL_MOISTURE] / (df["Soil_enc"] + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in correct order.
    Must be called after data_loader.encode_categoricals().

    Returns
    -------
    df with all original + engineered columns present.
    """
    df = add_npk_features(df)       # needs N, K, P columns
    df = add_temp_bins(df)          # needs COL_TEMP
    df = add_interaction_terms(df)  # needs COL_TEMP, COL_HUMIDITY, Soil_enc
    df = add_crop_soil_interaction(df)   # needs Crop_enc, Soil_enc
    df = add_stress_indices(df)     # needs COL_TEMP, COL_HUMIDITY, COL_MOISTURE
    return df


def get_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return THREE feature matrices for progressive comparison:
      X_raw      – 5 features: env + soil type + crop (no soil test)
      X_npk      – 8 features: + raw N, K, P soil-test values
      X_eng      – 21 features: + all derived domain features
    """
    X_raw = df[RAW_FEATURES].copy()
    X_npk = df[RAW_FEATURES_WITH_NPK].copy()
    X_eng = df[ENGINEERED_FEATURES].copy()
    return X_raw, X_npk, X_eng
