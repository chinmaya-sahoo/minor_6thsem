"""
data_loader.py
==============
Responsibilities
----------------
- Read the CSV
- Validate columns & missing values
- Encode categorical columns (Soil Type, Crop Type, Fertilizer Name)
- Return a clean DataFrame + fitted LabelEncoders for later inverse-transform
- Perform the 80-20 stratified train/test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import (
    DATA_PATH, TEST_SIZE, RANDOM_STATE,
    COL_SOIL, COL_CROP, COL_FERT,
)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """
    Read the CSV and return a clean DataFrame.
    Raises
    ------
    FileNotFoundError  – if the CSV does not exist
    ValueError         – if expected columns are missing
    """
    df = pd.read_csv(DATA_PATH)

    required = [COL_SOIL, COL_CROP, COL_FERT]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    n_missing = df.isnull().sum().sum()
    if n_missing:
        print(f"  ⚠  {n_missing} missing values found – dropping rows.")
        df = df.dropna().reset_index(drop=True)

    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode Soil Type, Crop Type, and Fertilizer Name.

    Parameters
    ----------
    df : raw DataFrame

    Returns
    -------
    df      : DataFrame with added *_enc columns
    encoders: dict  {"soil": le, "crop": le, "fert": le}
    """
    df = df.copy()
    encoders = {}

    col_out_map = {"soil": "Soil_enc", "crop": "Crop_enc", "fert": "Fert_enc"}

    for col, key in [(COL_SOIL, "soil"), (COL_CROP, "crop"), (COL_FERT, "fert")]:
        le = LabelEncoder()
        df[col_out_map[key]] = le.fit_transform(df[col])
        encoders[key] = le

    return df, encoders


def load_and_encode() -> tuple[pd.DataFrame, dict]:
    """Convenience: load + encode in one call."""
    df = load_raw()
    df, encoders = encode_categoricals(df)

    print(f"  ✔  Loaded  : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"     Crops   : {sorted(df[COL_CROP].unique())}")
    print(f"     Soils   : {sorted(df[COL_SOIL].unique())}")
    print(f"     Fertil. : {sorted(df[COL_FERT].unique())}")

    return df, encoders


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Stratified 80-20 train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"  ✔  Train : {len(X_train)} samples  ({100*(1-test_size):.0f}%)")
    print(f"     Test  : {len(X_test)} samples  ({100*test_size:.0f}%)")
    return X_train, X_test, y_train, y_test
