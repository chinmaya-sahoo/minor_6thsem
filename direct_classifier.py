"""
direct_classifier.py
====================
Direct Fertilizer Classification using Cleaned Dataset

Pipeline:
  Input : Temp + Humidity + Moisture + Soil Type + Crop Type
        + Nitrogen + Potassium + Phosphorous  (soil-test values)
  Output: Fertilizer Name  (7 classes)

Algorithms compared:
  1. Logistic Regression
  2. Ridge Classifier
  3. Decision Tree
  4. KNN Classifier
  5. Random Forest
  6. XGBoost

Metrics reported:
  Accuracy, F1, Precision, Recall, Cohen's Kappa, MCC

Charts generated (saved to results_direct/):
  01_accuracy_bar.png          – Accuracy of all algorithms
  02_all_metrics_bar.png       – All 6 metrics side by side
  03_confusion_matrix_*.png    – Confusion matrix per algorithm
  04_metrics_heatmap.png       – Metric comparison heatmap
  05_radar_chart.png           – Spider chart across all metrics
  06_feature_importance.png    – Top feature importances (RF + XGBoost)
  07_roc_curves.png            – One-vs-Rest ROC curves (top 2 models)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

from sklearn.preprocessing   import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, auc,
)
from sklearn.linear_model    import LogisticRegression, RidgeClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.ensemble        import RandomForestClassifier
from xgboost                 import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_PATH   = os.path.join("data", "data_core_cleaned.csv")
RESULTS_DIR  = "results_direct"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

BG_DARK  = "#0d0d1a"
BG_PANEL = "#1a1a2e"
GRID_CLR = "#2a2a4a"
TEXT_CLR = "#e0e0ff"
AXIS_CLR = "#444466"

ALGO_COLORS = {
    "Logistic Regression": "#4361EE",
    "Ridge Classifier"   : "#3A86FF",
    "Decision Tree"      : "#F72585",
    "KNN Classifier"     : "#FF9F1C",
    "Random Forest"      : "#06D6A0",
    "XGBoost"            : "#7209B7",
}
SHORT = ["LR", "Ridge", "DT", "KNN", "RF", "XGB"]

os.makedirs(RESULTS_DIR, exist_ok=True)
plt.style.use("dark_background")

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║   DIRECT FERTILIZER CLASSIFIER  ·  Cleaned Dataset                  ║
║   Input  : Env + Soil + Crop + Soil-Test NPK  (15 features)         ║
║   Output : Fertilizer Name  (7 classes)                             ║
╚══════════════════════════════════════════════════════════════════════╝
"""
print(BANNER)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Encode
# ─────────────────────────────────────────────────────────────────────────────
print("── Step 1 · Load Cleaned Dataset ─────────────────────────────────────")
if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(
        f"{CLEAN_PATH} not found.\n"
        "Run  python3 main.py  once first to generate the cleaned CSV."
    )

df = pd.read_csv(CLEAN_PATH)
print(f"  ✔  Rows: {len(df)}   Cols: {df.shape[1]}")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df["Soil_enc"] = le_soil.fit_transform(df["Soil Type"])
df["Crop_enc"] = le_crop.fit_transform(df["Crop Type"])
df["Fert_enc"] = le_fert.fit_transform(df["Fertilizer Name"])

FERT_LABELS = list(le_fert.classes_)
print(f"  ✔  Classes  : {FERT_LABELS}")
print(f"  ✔  Crops    : {list(le_crop.classes_)}")
print(f"  ✔  Soils    : {list(le_soil.classes_)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 2 · Feature Engineering ────────────────────────────────────────")

# NPK-derived features
df["NPK_total"]    = df["Nitrogen"] + df["Potassium"] + df["Phosphorous"]
df["NPK_N_ratio"]  = df["Nitrogen"]    / (df["NPK_total"] + 1)
df["NPK_K_ratio"]  = df["Potassium"]   / (df["NPK_total"] + 1)
df["NPK_P_ratio"]  = df["Phosphorous"] / (df["NPK_total"] + 1)
df["NPK_imbalance"]= df[["Nitrogen","Potassium","Phosphorous"]].max(axis=1) \
                   / (df[["Nitrogen","Potassium","Phosphorous"]].min(axis=1) + 1)
df["NPK_dominant"] = df[["Nitrogen","Potassium","Phosphorous"]].values.argmax(axis=1)

# Temperature bins
df["Temp_bin"] = pd.cut(
    df["Temparature"], bins=[0, 25, 31, 36, 100],
    labels=[0, 1, 2, 3], right=False,
).astype(int)

# Interaction terms
n_soil = df["Soil_enc"].max() + 1
df["Temp_x_Moisture"]   = df["Temparature"] * df["Moisture"]
df["Humidity_x_Soil"]   = df["Humidity"]    * df["Soil_enc"]
df["Temp_x_Humidity"]   = df["Temparature"] * df["Humidity"]
df["Crop_Soil_interact"] = df["Crop_enc"] * n_soil + df["Soil_enc"]
df["Heat_Stress_idx"]    = df["Temparature"] / (df["Humidity"] + 1)
df["Moisture_Soil_idx"]  = df["Moisture"]   / (df["Soil_enc"] + 1)

FEATURES = [
    # Raw env + soil + crop
    "Temparature", "Humidity", "Moisture", "Soil_enc", "Crop_enc",
    # Soil-test NPK readings
    "Nitrogen", "Potassium", "Phosphorous",
    # NPK-derived
    "NPK_total", "NPK_N_ratio", "NPK_K_ratio", "NPK_P_ratio",
    "NPK_imbalance", "NPK_dominant",
    # Env engineering
    "Temp_bin",
    "Temp_x_Moisture", "Humidity_x_Soil", "Temp_x_Humidity",
    "Crop_Soil_interact", "Heat_Stress_idx", "Moisture_Soil_idx",
]

print(f"  ✔  Total features: {len(FEATURES)}")
print(f"  ✔  {FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Split
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 3 · 80-20 Stratified Split ─────────────────────────────────────")

X = df[FEATURES].values
y = df["Fert_enc"].values

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Scale once — each classifier will use the same scaled data
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

print(f"  ✔  Train : {len(y_tr)}   Test : {len(y_te)}")
print(f"  ✔  Class distribution (test): "
      + ", ".join(f"{lb}={np.sum(y_te==i)}" for i, lb in enumerate(FERT_LABELS)))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Model Definitions
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True
    ),
    "Ridge Classifier": (
        RidgeClassifier(), True
    ),
    "Decision Tree": (
        DecisionTreeClassifier(
            max_depth=20, min_samples_leaf=2, random_state=RANDOM_STATE
        ), False
    ),
    "KNN Classifier": (
        KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1), True
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1
        ), False
    ),
    "XGBoost": (
        XGBClassifier(
            n_estimators=400, max_depth=7, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            use_label_encoder=False, eval_metric="mlogloss",
        ), False
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 4 · Training ────────────────────────────────────────────────────")

results   = {}
algo_names = list(MODELS.keys())

for name, (model, use_scaled) in MODELS.items():
    Xtr = X_tr_s if use_scaled else X_tr
    Xte = X_te_s if use_scaled else X_te

    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="weighted")
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted")
    kap  = cohen_kappa_score(y_te, y_pred)
    mcc  = matthews_corrcoef(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)

    results[name] = {
        "model" : model,
        "y_pred": y_pred,
        "Accuracy" : acc,
        "F1"       : f1,
        "Precision": prec,
        "Recall"   : rec,
        "Kappa"    : kap,
        "MCC"      : mcc,
        "CM"       : cm,
        "scaled"   : use_scaled,
    }

    print(f"\n  [{name}]")
    print(f"    Accuracy  = {acc:.4f}")
    print(f"    F1        = {f1:.4f}")
    print(f"    Precision = {prec:.4f}")
    print(f"    Recall    = {rec:.4f}")
    print(f"    Kappa     = {kap:.4f}")
    print(f"    MCC       = {mcc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary Table + Classification Report
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 5 · Comparison Table ────────────────────────────────────────────\n")
METRIC_KEYS = ["Accuracy", "F1", "Precision", "Recall", "Kappa", "MCC"]
header = f"  {'Algorithm':<22}" + "".join(f"{m:>12}" for m in METRIC_KEYS)
print(header)
print("  " + "─" * (22 + 12 * len(METRIC_KEYS)))

for name, res in results.items():
    row = f"  {name:<22}" + "".join(f"{res[k]:>12.4f}" for k in METRIC_KEYS)
    print(row)

best = max(algo_names, key=lambda n: results[n]["Accuracy"])
print(f"\n  🏆  Best Model : {best}  ({results[best]['Accuracy']*100:.2f}%)")

print(f"\n── Classification Report [{best}] ─────────────────────────────────────")
print(classification_report(y_te, results[best]["y_pred"],
                             target_names=FERT_LABELS))

# ─────────────────────────────────────────────────────────────────────────────
# 7. Charts
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 6 · Generating Charts ───────────────────────────────────────────")

colors = [ALGO_COLORS[n] for n in algo_names]

def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Saved: {path}")


# ── Chart 1 : Accuracy bar ──────────────────────────────────────────────────
accs  = [results[n]["Accuracy"] * 100 for n in algo_names]
order = np.argsort(accs)

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_PANEL)

bars = ax.barh([algo_names[i] for i in order],
               [accs[i] for i in order],
               color=[colors[i] for i in order],
               edgecolor="#111", linewidth=0.5, height=0.6, zorder=3)
for bar in bars:
    v = bar.get_width()
    ax.text(v + 0.2, bar.get_y() + bar.get_height()/2,
            f"{v:.2f}%", va="center", fontsize=10,
            color=TEXT_CLR, fontweight="bold")

ax.set_title("🏆  Fertilizer Classification Accuracy — Direct Classifier",
             fontsize=13, fontweight="bold", color=TEXT_CLR, pad=10)
ax.set_xlabel("Accuracy %", color="#aaaacc", fontsize=10)
ax.set_xlim(0, 115)
ax.tick_params(colors=TEXT_CLR, labelsize=10)
ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5, alpha=0.7, zorder=0)
plt.tight_layout()
_save(fig, "01_accuracy_bar.png")


# ── Chart 2 : All 6 metrics grouped bar ─────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("📊  All Metrics — Direct Fertilizer Classifier",
             fontsize=15, fontweight="bold", color=TEXT_CLR, y=1.01)

x = np.arange(len(algo_names))
for ax, mk in zip(axes.flat, METRIC_KEYS):
    vals = [results[n][mk] for n in algo_names]
    bars = ax.bar(x, vals, color=colors, edgecolor="#111", linewidth=0.5,
                  width=0.6, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.005,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT_CLR, fontweight="bold")
    ax.set_title(mk, fontsize=12, fontweight="bold", color=TEXT_CLR)
    ax.set_xticks(x);  ax.set_xticklabels(SHORT, fontsize=9, color=TEXT_CLR)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors=TEXT_CLR)
    ax.set_facecolor(BG_PANEL)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)

plt.tight_layout()
_save(fig, "02_all_metrics_bar.png")


# ── Chart 3 : Confusion matrices (all 6) ────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("🔢  Confusion Matrices — All Algorithms",
             fontsize=15, fontweight="bold", color=TEXT_CLR, y=1.01)

cmap_cm = LinearSegmentedColormap.from_list(
    "cm", ["#0d0d1a", "#4361EE", "#06D6A0", "#FFD60A"], N=256
)

for ax, name in zip(axes.flat, algo_names):
    cm = results[name]["CM"].astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1)
    for i in range(len(FERT_LABELS)):
        for j in range(len(FERT_LABELS)):
            clr = "black" if cm_norm[i, j] > 0.6 else "white"
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=clr)
    ax.set_title(f"{name}  (Acc={results[name]['Accuracy']*100:.1f}%)",
                 fontsize=10, fontweight="bold", color=TEXT_CLR)
    ax.set_xticks(range(len(FERT_LABELS)))
    ax.set_yticks(range(len(FERT_LABELS)))
    ax.set_xticklabels(FERT_LABELS, rotation=35, ha="right",
                       fontsize=8, color=TEXT_CLR)
    ax.set_yticklabels(FERT_LABELS, fontsize=8, color=TEXT_CLR)
    ax.set_xlabel("Predicted", fontsize=9, color="#aaaacc")
    ax.set_ylabel("Actual",    fontsize=9, color="#aaaacc")
    ax.tick_params(colors=TEXT_CLR)
    ax.spines[:].set_color(AXIS_CLR)

plt.tight_layout()
_save(fig, "03_confusion_matrices.png")


# ── Chart 4 : Metrics heatmap ───────────────────────────────────────────────
mat = np.array([[results[n][k] for n in algo_names] for k in METRIC_KEYS])
mat_norm = (mat - mat.min(axis=1, keepdims=True)) / \
           (mat.max(axis=1, keepdims=True) - mat.min(axis=1, keepdims=True) + 1e-9)

cmap_hm = LinearSegmentedColormap.from_list(
    "hm", ["#0d0d1a", "#4361EE", "#06D6A0", "#FFD60A"], N=256
)

fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_DARK)
im = ax.imshow(mat_norm, cmap=cmap_hm, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(algo_names)))
ax.set_xticklabels(algo_names, rotation=18, ha="right",
                   fontsize=10, color=TEXT_CLR)
ax.set_yticks(range(len(METRIC_KEYS)))
ax.set_yticklabels(METRIC_KEYS, fontsize=11, color=TEXT_CLR)
ax.set_title("📈  Metrics Heatmap  (brighter = better)",
             fontsize=13, fontweight="bold", color=TEXT_CLR, pad=12)
ax.tick_params(colors=TEXT_CLR)
ax.spines[:].set_color(AXIS_CLR)

for i, mk in enumerate(METRIC_KEYS):
    for j, name in enumerate(algo_names):
        v = mat[i, j]
        ax.text(j, i, f"{v:.4f}",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="black" if mat_norm[i, j] > 0.6 else "white")

cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.ax.tick_params(colors=TEXT_CLR)
cbar.set_label("Normalised score", color=TEXT_CLR, fontsize=9)
plt.tight_layout()
_save(fig, "04_metrics_heatmap.png")


# ── Chart 5 : Radar chart ───────────────────────────────────────────────────
N_METRICS = len(METRIC_KEYS)
angles = np.linspace(0, 2 * np.pi, N_METRICS, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_PANEL)

for name, color in ALGO_COLORS.items():
    vals = [results[name][k] for k in METRIC_KEYS]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, color=color, label=name)
    ax.fill(angles, vals, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(METRIC_KEYS, fontsize=11, color=TEXT_CLR)
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"],
                   fontsize=8, color="#666688")
ax.grid(color=GRID_CLR, linewidth=0.8)
ax.spines["polar"].set_color(AXIS_CLR)
ax.set_title("🕸️  Metric Radar — All Algorithms",
             fontsize=14, fontweight="bold", color=TEXT_CLR, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
          facecolor=BG_PANEL, edgecolor=AXIS_CLR,
          labelcolor=TEXT_CLR, fontsize=10)
plt.tight_layout()
_save(fig, "05_radar_chart.png")


# ── Chart 6 : Feature Importance (RF + XGBoost) ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("🌲  Feature Importances",
             fontsize=14, fontweight="bold", color=TEXT_CLR)

TOP_N = 15
for ax, model_name, color in [
    (ax1, "Random Forest", "#06D6A0"),
    (ax2, "XGBoost",       "#7209B7"),
]:
    model = results[model_name]["model"]
    imps  = model.feature_importances_
    order = np.argsort(imps)[-TOP_N:]
    bars  = ax.barh(
        [FEATURES[i] for i in order],
        [imps[i]    for i in order],
        color=color, edgecolor="#111", linewidth=0.5, height=0.7, zorder=3,
    )
    for bar in bars:
        ax.text(bar.get_width() + 0.001,
                bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.4f}",
                va="center", fontsize=8, color=TEXT_CLR, fontweight="bold")
    ax.set_title(model_name, fontsize=12, fontweight="bold", color=TEXT_CLR)
    ax.set_xlabel("Importance", fontsize=10, color="#aaaacc")
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5, alpha=0.7)

plt.tight_layout()
_save(fig, "06_feature_importance.png")


# ── Chart 7 : ROC curves (top 2 models, One-vs-Rest) ────────────────────────
TOP2 = sorted(algo_names, key=lambda n: results[n]["Accuracy"], reverse=True)[:2]

# Only models with predict_proba support ROC
proba_models = {
    "Logistic Regression": True,
    "Ridge Classifier"   : False,   # no predict_proba
    "Decision Tree"      : True,
    "KNN Classifier"     : True,
    "Random Forest"      : True,
    "XGBoost"            : True,
}

y_bin = label_binarize(y_te, classes=list(range(len(FERT_LABELS))))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("📉  One-vs-Rest ROC Curves — Top 2 Models",
             fontsize=14, fontweight="bold", color=TEXT_CLR)

fert_palette = ["#4361EE","#F72585","#06D6A0","#FF9F1C","#7209B7","#3A86FF","#FFD60A"]

for ax, name in zip(axes, TOP2):
    ax.set_facecolor(BG_PANEL)
    model = results[name]["model"]
    Xte_used = X_te_s if results[name]["scaled"] else X_te

    if proba_models.get(name, False) and hasattr(model, "predict_proba"):
        y_score = model.predict_proba(Xte_used)
    else:
        ax.text(0.5, 0.5, f"{name}\ndoes not support\npredict_proba",
                ha="center", va="center", fontsize=14, color=TEXT_CLR,
                transform=ax.transAxes)
        ax.set_title(name, fontsize=11, fontweight="bold", color=TEXT_CLR)
        continue

    for i, (fert, fcolor) in enumerate(zip(FERT_LABELS, fert_palette)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=fcolor, linewidth=2,
                label=f"{fert}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", linewidth=1, alpha=0.5)
    ax.set_title(f"{name}  (Acc={results[name]['Accuracy']*100:.1f}%)",
                 fontsize=12, fontweight="bold", color=TEXT_CLR)
    ax.set_xlabel("False Positive Rate", fontsize=10, color="#aaaacc")
    ax.set_ylabel("True Positive Rate",  fontsize=10, color="#aaaacc")
    ax.tick_params(colors=TEXT_CLR)
    ax.spines["top"].set_visible(False);   ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=8.5, facecolor=BG_PANEL, edgecolor=AXIS_CLR,
              labelcolor=TEXT_CLR, loc="lower right")

plt.tight_layout()
_save(fig, "07_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Final Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Final Leaderboard ────────────────────────────────────────────────────\n")
leaderboard = sorted(algo_names,
                     key=lambda n: results[n]["Accuracy"], reverse=True)

print(f"  {'Rank':<5} {'Algorithm':<22}"
      + "".join(f"{m:>12}" for m in METRIC_KEYS))
print("  " + "─" * (5 + 22 + 12 * len(METRIC_KEYS)))

medals = ["🥇", "🥈", "🥉", "4 ", "5 ", "6 "]
for rank, name in enumerate(leaderboard):
    row = f"  {medals[rank]}  {name:<22}"
    for mk in METRIC_KEYS:
        row += f"{results[name][mk]:>12.4f}"
    print(row)

print(f"\n  🏆  Champion : {leaderboard[0]}  "
      f"— Accuracy {results[leaderboard[0]]['Accuracy']*100:.2f}%"
      f"  |  F1 {results[leaderboard[0]]['F1']:.4f}"
      f"  |  Kappa {results[leaderboard[0]]['Kappa']:.4f}")
print(f"\n  📁  All charts saved to : {os.path.abspath(RESULTS_DIR)}")
print("═" * 70 + "\n")
