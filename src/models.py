"""
models.py
=========
Single registry of all classifier definitions used in the pipeline.

Why a registry?
---------------
- Adding/removing a model requires a change in ONE place only.
- trainer.py, visualizer.py, and main.py never hard-code model names.
- Each model can declare whether it needs scaled input features (needs_scale).
"""

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────
# Each entry:  "Display Name" → {"model": estimator, "needs_scale": bool}
# needs_scale=True  → StandardScaler will be applied before fit/predict

MODEL_REGISTRY: dict[str, dict] = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "needs_scale": True,
    },

    "Ridge Classifier": {
        "model": RidgeClassifier(alpha=1.0),
        "needs_scale": True,
    },

    "Decision Tree": {
        "model": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "needs_scale": False,
    },

    "KNN Classifier": {
        "model": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
        "needs_scale": True,    # KNN is distance-based → must scale
    },

    "Random Forest": {
        "model": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "needs_scale": False,
    },

    "XGBoost": {
        "model": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
        ),
        "needs_scale": False,
    },
}


def get_model_names() -> list[str]:
    """Return all registered model display names."""
    return list(MODEL_REGISTRY.keys())


def get_model(name: str):
    """Return a FRESH clone of the model (avoids state leakage between runs)."""
    from sklearn.base import clone
    entry = MODEL_REGISTRY[name]
    return clone(entry["model"]), entry["needs_scale"]
