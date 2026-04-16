"""
config.py
=========
Central configuration: paths, splits, random seeds, algorithm colours.
Changing a value here propagates everywhere in the pipeline.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH        = os.path.join(BASE_DIR, "data", "data_core.csv")
CLEAN_DATA_PATH  = os.path.join(BASE_DIR, "data", "data_core_cleaned.csv")   # saved after correction
RESULTS_DIR      = os.path.join(BASE_DIR, "results_v2")

# ── Split ─────────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Column names (as they appear in the CSV) ──────────────────────────────────
COL_TEMP     = "Temparature"        # note: original typo kept
COL_HUMIDITY = "Humidity"
COL_MOISTURE = "Moisture"
COL_SOIL     = "Soil Type"
COL_CROP     = "Crop Type"
COL_N        = "Nitrogen"
COL_K        = "Potassium"
COL_P        = "Phosphorous"
COL_FERT     = "Fertilizer Name"

# ── Feature groups ────────────────────────────────────────────────────────────
# Baseline: only environment + soil type + crop type  (no soil test)
RAW_FEATURES = [COL_TEMP, COL_HUMIDITY, COL_MOISTURE, "Soil_enc", "Crop_enc"]

# Corrected baseline: env + soil type + crop + soil-test NPK readings
# (In practice a farmer gets N,P,K from a soil lab before buying fertilizer)
RAW_FEATURES_WITH_NPK = [
    COL_TEMP, COL_HUMIDITY, COL_MOISTURE,
    "Soil_enc", "Crop_enc",
    COL_N, COL_K, COL_P,
]

# Full engineered set: all of the above + derived domain features
ENGINEERED_FEATURES = [
    # environment + soil + crop
    COL_TEMP, COL_HUMIDITY, COL_MOISTURE, "Soil_enc", "Crop_enc",
    # soil-test NPK readings
    COL_N, COL_K, COL_P,
    # NPK-derived features
    "NPK_total",          # total macro-nutrient load
    "NPK_N_ratio",        # nitrogen share  (tells model if N-heavy fert needed)
    "NPK_K_ratio",        # potassium share
    "NPK_P_ratio",        # phosphorous share
    "NPK_imbalance",      # max/min ratio → how skewed the NPK profile is
    "NPK_dominant",       # which nutrient dominates (0=N, 1=K, 2=P)
    # temperature bins (growing season)
    "Temp_bin",
    # environment interaction terms
    "Temp_x_Moisture",
    "Humidity_x_Soil",
    "Temp_x_Humidity",
    # crop-soil compatibility
    "Crop_Soil_interact",
    # stress & deficiency proxies
    "Heat_Stress_idx",
    "Moisture_Soil_idx",
]

TARGET = "Fert_enc"

# ── Algorithm colour palette ──────────────────────────────────────────────────
ALGO_COLORS = {
    "Logistic Regression" : "#4361EE",
    "Ridge Classifier"    : "#3A86FF",
    "Decision Tree"       : "#F72585",
    "KNN Classifier"      : "#FF9F1C",
    "Random Forest"       : "#06D6A0",
    "XGBoost"             : "#7209B7",
}

ALGO_NAMES  = list(ALGO_COLORS.keys())
SHORT_NAMES = ["LR", "Ridge", "DT", "KNN", "RF", "XGB"]
