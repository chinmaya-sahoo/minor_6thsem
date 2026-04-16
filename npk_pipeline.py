"""
npk_pipeline.py
===============
Two-Layer Fertilizer Prediction — Cleaned Dataset Only

Architecture
------------
Layer 1  (Regression)     : (Env + Soil + Crop) → predict N, K, P
Layer 2  (Rule-based)     : (Predicted N, K, P)  → Fertilizer Name

Why rule-based for Layer 2?
    The cleaned dataset's fertilizer labels were derived from an NPK
    domain rule, so using the same rule for Layer 2 is perfectly consistent
    and avoids compounding two sources of model error.

Metrics compared (per algorithm)
---------------------------------
  N R²          – R² for Nitrogen   prediction
  P R²          – R² for Phosphorous prediction
  K R²          – R² for Potassium  prediction
  Overall R²    – mean of N, P, K R²
  Avg RMSE      – mean RMSE across N, P, K
  Avg MAE       – mean MAE  across N, P, K
  Fertilizer %  – how often the rule gives the correct fertilizer
                  when given the PREDICTED N, K, P values

Algorithms: Linear Regression · Ridge · Decision Tree ·
            KNN · Random Forest · XGBoost
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

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base            import clone

from sklearn.linear_model  import LinearRegression, Ridge
from sklearn.tree          import DecisionTreeRegressor
from sklearn.neighbors     import KNeighborsRegressor
from sklearn.ensemble      import RandomForestRegressor
from xgboost               import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_PATH   = os.path.join("data", "data_core_cleaned.csv")
RESULTS_DIR  = "results_npk"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Encode
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 1 · Load Cleaned Dataset ─────────────────────────────────────")
if not os.path.exists(CLEAN_PATH):
    raise FileNotFoundError(
        f"{CLEAN_PATH} not found.\n"
        "Run  python3 main.py  once first to generate the cleaned CSV."
    )

df = pd.read_csv(CLEAN_PATH)
print(f"  ✔  Rows: {len(df)}  Cols: {df.shape[1]}")

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df["Soil_enc"] = le_soil.fit_transform(df["Soil Type"])
df["Crop_enc"] = le_crop.fit_transform(df["Crop Type"])
df["Fert_enc"] = le_fert.fit_transform(df["Fertilizer Name"])

FERT_LABELS = list(le_fert.classes_)
print(f"  ✔  Fertilizers: {FERT_LABELS}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Engineering  (env + soil + crop — NOT N/K/P, those are targets)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 2 · Feature Engineering ────────────────────────────────────────")

# Temperature growing-season bins
df["Temp_bin"] = pd.cut(
    df["Temparature"], bins=[0, 25, 31, 36, 100],
    labels=[0, 1, 2, 3], right=False,
).astype(int)

# Pairwise interactions
df["Temp_x_Moisture"]  = df["Temparature"] * df["Moisture"]
df["Humidity_x_Soil"]  = df["Humidity"]    * df["Soil_enc"]
df["Temp_x_Humidity"]  = df["Temparature"] * df["Humidity"]

# Crop-Soil compatibility (single integer encoding the pair)
n_soil = df["Soil_enc"].max() + 1
df["Crop_Soil_interact"] = df["Crop_enc"] * n_soil + df["Soil_enc"]

# Environmental stress proxies
df["Heat_Stress_idx"]   = df["Temparature"] / (df["Humidity"]  + 1)
df["Moisture_Soil_idx"] = df["Moisture"]    / (df["Soil_enc"]  + 1)

FEATURES = [
    "Temparature", "Humidity", "Moisture",
    "Soil_enc", "Crop_enc",
    "Temp_bin",
    "Temp_x_Moisture", "Humidity_x_Soil", "Temp_x_Humidity",
    "Crop_Soil_interact",
    "Heat_Stress_idx", "Moisture_Soil_idx",
]

print(f"  ✔  Input features ({len(FEATURES)}): {FEATURES}")
print(f"     Targets (regression): Nitrogen, Potassium, Phosphorous")
print(f"     Target  (final)     : Fertilizer Name")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Split
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 3 · 80-20 Stratified Split ─────────────────────────────────────")

X = df[FEATURES].values
y_N    = df["Nitrogen"].values.astype(float)
y_K    = df["Potassium"].values.astype(float)
y_P    = df["Phosphorous"].values.astype(float)
y_fert = df["Fert_enc"].values

# Stratified split on fertilizer label for balanced classes
X_tr, X_te, \
yN_tr, yN_te, \
yK_tr, yK_te, \
yP_tr, yP_te, \
yf_tr, yf_te   = [None]*10

(X_tr, X_te,
 yN_tr, yN_te) = train_test_split(
    X, y_N, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    stratify=y_fert,
)
# Use same index for K, P, fert
idx = np.arange(len(df))
idx_tr, idx_te = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_fert,
)

X_tr  = df.iloc[idx_tr][FEATURES].values
X_te  = df.iloc[idx_te][FEATURES].values
yN_tr = df.iloc[idx_tr]["Nitrogen"].values.astype(float)
yN_te = df.iloc[idx_te]["Nitrogen"].values.astype(float)
yK_tr = df.iloc[idx_tr]["Potassium"].values.astype(float)
yK_te = df.iloc[idx_te]["Potassium"].values.astype(float)
yP_tr = df.iloc[idx_tr]["Phosphorous"].values.astype(float)
yP_te = df.iloc[idx_te]["Phosphorous"].values.astype(float)
fert_true_names = df.iloc[idx_te]["Fertilizer Name"].values

print(f"  ✔  Train: {len(idx_tr)}   Test: {len(idx_te)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NPK Rule  (same rule used to build the cleaned dataset)
# ─────────────────────────────────────────────────────────────────────────────

def npk_rule(N: float, K: float, P: float) -> str:
    """Convert predicted N, K, P requirements into a fertilizer recommendation."""
    if K < 3:
        if N >= 30 and P < 5:           return "Urea"
        elif P >= 30:                    return "DAP"
        elif N >= 18 and P >= 14 and N >= P: return "28-28"
        else:                            return "20-20"
    balanced   = abs(N-K) <= 7 and abs(K-P) <= 7 and abs(N-P) <= 7
    k_p_close  = abs(K-P) < max(K, 1) * 0.45
    both_dom_n = K > N * 1.4 and P > N * 1.4
    p_dom_k    = P > K * 1.5 and P > N
    if balanced and not (k_p_close and both_dom_n): return "17-17-17"
    elif k_p_close and both_dom_n:                  return "10-26-26"
    elif p_dom_k:                                   return "14-35-14"
    else:                                           return "17-17-17"

# ─────────────────────────────────────────────────────────────────────────────
# 5. Model Registry
# ─────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "Linear Regression": (
        LinearRegression(), False
    ),
    "Ridge Regression": (
        Ridge(alpha=1.0), False
    ),
    "Decision Tree": (
        DecisionTreeRegressor(
            max_depth=15, min_samples_leaf=3, random_state=RANDOM_STATE,
        ), False
    ),
    "KNN Regressor": (
        KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1),
        True,                           # needs StandardScaler
    ),
    "Random Forest": (
        RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ), False
    ),
    "XGBoost": (
        XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        ), False
    ),
}

ALGO_COLORS = {
    "Linear Regression": "#4361EE",
    "Ridge Regression" : "#3A86FF",
    "Decision Tree"    : "#F72585",
    "KNN Regressor"    : "#FF9F1C",
    "Random Forest"    : "#06D6A0",
    "XGBoost"          : "#7209B7",
}

# ─────────────────────────────────────────────────────────────────────────────
# Also run DIRECT classification (all features incl. true N, K, P → Fertilizer)
# gives the "ceiling" accuracy achievable when a soil-test is available
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.linear_model  import LogisticRegression, RidgeClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.ensemble      import RandomForestClassifier
from xgboost               import XGBClassifier

ALL_FEATURES = FEATURES + ["Nitrogen", "Potassium", "Phosphorous"]

X_all_tr = df.iloc[idx_tr][ALL_FEATURES].values
X_all_te = df.iloc[idx_te][ALL_FEATURES].values
y_fert_tr = df.iloc[idx_tr]["Fert_enc"].values
y_fert_te = df.iloc[idx_te]["Fert_enc"].values

CLASSIFIERS = {
    "Linear Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Ridge Regression" : RidgeClassifier(),
    "Decision Tree"    : DecisionTreeClassifier(max_depth=15, random_state=RANDOM_STATE),
    "KNN Regressor"    : KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
    "Random Forest"    : RandomForestClassifier(n_estimators=200, max_depth=15,
                                                random_state=RANDOM_STATE, n_jobs=-1),
    "XGBoost"          : XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.08,
                                        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1),
}

print("\n── Direct Classifier (Env + Soil + Crop + Soil-Test NPK → Fertilizer) ──")
DIRECT_ACC = {}
scaler_direct = StandardScaler()
X_d_tr = scaler_direct.fit_transform(X_all_tr)
X_d_te = scaler_direct.transform(X_all_te)

for cname, clf in CLASSIFIERS.items():
    clf.fit(X_d_tr, y_fert_tr)
    acc = (clf.predict(X_d_te) == y_fert_te).mean()
    DIRECT_ACC[cname] = acc
    print(f"    {cname:<22}  Direct Accuracy = {acc*100:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Train & Evaluate
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 4 · Training and Evaluating Algorithms ─────────────────────────")

results = {}

for algo_name, (base_model, needs_scale) in MODEL_REGISTRY.items():

    # Optionally scale
    scaler = None
    Xtr, Xte = X_tr, X_te
    if needs_scale:
        scaler = StandardScaler()
        Xtr    = scaler.fit_transform(X_tr)
        Xte    = scaler.transform(X_te)

    # Train three independent regressors (one per nutrient)
    reg_N = clone(base_model)
    reg_K = clone(base_model)
    reg_P = clone(base_model)

    reg_N.fit(Xtr, yN_tr)
    reg_K.fit(Xtr, yK_tr)
    reg_P.fit(Xtr, yP_tr)

    # Predictions
    pN = reg_N.predict(Xte)
    pK = reg_K.predict(Xte)
    pP = reg_P.predict(Xte)

    # Clamp negatives (physical impossibility)
    pN = np.clip(pN, 0, None)
    pK = np.clip(pK, 0, None)
    pP = np.clip(pP, 0, None)

    # ── Regression metrics ───────────────────────────────────────────────
    r2_N  = r2_score(yN_te, pN)
    r2_K  = r2_score(yK_te, pK)
    r2_P  = r2_score(yP_te, pP)
    overall_r2 = (r2_N + r2_K + r2_P) / 3

    rmse_N = mean_squared_error(yN_te, pN) ** 0.5
    rmse_K = mean_squared_error(yK_te, pK) ** 0.5
    rmse_P = mean_squared_error(yP_te, pP) ** 0.5
    avg_rmse = (rmse_N + rmse_K + rmse_P) / 3

    mae_N  = mean_absolute_error(yN_te, pN)
    mae_K  = mean_absolute_error(yK_te, pK)
    mae_P  = mean_absolute_error(yP_te, pP)
    avg_mae = (mae_N + mae_K + mae_P) / 3

    # ── Layer 2 : NPK rule → Fertilizer ─────────────────────────────────
    pred_fert = np.array([npk_rule(n, k, p) for n, k, p in zip(pN, pK, pP)])
    fert_match = np.mean(pred_fert == fert_true_names)

    results[algo_name] = {
        "N_R2"      : r2_N,
        "K_R2"      : r2_K,
        "P_R2"      : r2_P,
        "Overall_R2": overall_r2,
        "Avg_RMSE"  : avg_rmse,
        "Avg_MAE"   : avg_mae,
        "Fert_Match": fert_match,
        # store raw predictions for charting
        "pN": pN, "pK": pK, "pP": pP,
        "pred_fert": pred_fert,
    }

    print(f"\n  [{algo_name}]")
    print(f"    N R²={r2_N:+.4f}  K R²={r2_K:+.4f}  P R²={r2_P:+.4f}  "
          f"Overall R²={overall_r2:+.4f}")
    print(f"    Avg RMSE={avg_rmse:.4f}   Avg MAE={avg_mae:.4f}")
    print(f"    Fertilizer Match = {fert_match*100:.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary Table
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 5 · Comparison Table ────────────────────────────────────────────\n")

COLS = ["N R²", "K R²", "P R²", "Overall R²", "Avg RMSE", "Avg MAE", "Fert %"]
KEY  = ["N_R2","K_R2","P_R2","Overall_R2","Avg_RMSE","Avg_MAE","Fert_Match"]

header = f"  {'Algorithm':<22}" + "".join(f"{c:>12}" for c in COLS)
print(header)
print("  " + "─" * (22 + 12 * len(COLS)))

for name, res in results.items():
    row = f"  {name:<22}"
    for k in KEY:
        v = res[k]
        if k == "Fert_Match":
            row += f"{v*100:>11.2f}%"
        else:
            row += f"{v:>12.4f}"
    print(row)

best_fert = max(results, key=lambda n: results[n]["Fert_Match"])
best_r2   = max(results, key=lambda n: results[n]["Overall_R2"])
print(f"\n  🏆  Best Fertilizer Match : {best_fert}  "
      f"({results[best_fert]['Fert_Match']*100:.2f}%)")
print(f"  🏆  Best Overall R²       : {best_r2}  "
      f"({results[best_r2]['Overall_R2']:.4f})")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Charts
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Step 6 · Generating Charts ───────────────────────────────────────────")

BG_DARK  = "#0d0d1a"
BG_PANEL = "#1a1a2e"
GRID_CLR = "#2a2a4a"
TEXT_CLR = "#e0e0ff"
AXIS_CLR = "#444466"
plt.style.use("dark_background")

algo_names  = list(results.keys())
algo_colors = [ALGO_COLORS[n] for n in algo_names]

def _save(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Saved: {path}")
    return path


# ── Chart 1 : All metrics grouped bar ──────────────────────────────────────
metric_keys   = ["N_R2", "K_R2", "P_R2", "Overall_R2"]
metric_labels = ["N R²", "K R²", "P R²", "Overall R²"]

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("🌿  NPK Regression R² per Algorithm — Cleaned Dataset",
             fontsize=14, fontweight="bold", color=TEXT_CLR, y=1.01)

x = np.arange(len(algo_names))
for i, (mk, ml) in enumerate(zip(metric_keys, metric_labels)):
    ax  = axes[i]
    vals = [results[n][mk] for n in algo_names]
    bars = ax.bar(x, vals, color=algo_colors, edgecolor="#111",
                  linewidth=0.5, width=0.55, zorder=3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                max(0, val) + 0.01,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")
    ax.set_title(ml, fontsize=11, fontweight="bold", color=TEXT_CLR)
    ax.set_xticks(x)
    ax.set_xticklabels(["LR","Ridge","DT","KNN","RF","XGB"],
                       fontsize=9, color=TEXT_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.set_facecolor(BG_PANEL)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_CLR)
    ax.spines["left"].set_color(AXIS_CLR)

plt.tight_layout()
_save(fig, "01_r2_comparison.png")


# ── Chart 2 : RMSE & MAE side by side ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle("📉  Error Metrics — Avg RMSE & Avg MAE",
             fontsize=13, fontweight="bold", color=TEXT_CLR)

for ax, mk, ml in [(ax1,"Avg_RMSE","Avg RMSE"), (ax2,"Avg_MAE","Avg MAE")]:
    vals  = [results[n][mk] for n in algo_names]
    order = np.argsort(vals)
    bars  = ax.barh(
        [algo_names[i] for i in order],
        [vals[i] for i in order],
        color=[algo_colors[i] for i in order],
        edgecolor="#111", linewidth=0.5, height=0.55, zorder=3,
    )
    for bar, val in zip(bars, sorted(vals)):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_title(ml, fontsize=11, fontweight="bold", color=TEXT_CLR)
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_CLR)
    ax.spines["left"].set_color(AXIS_CLR)
    ax.grid(axis="x", color=GRID_CLR, linewidth=0.5, alpha=0.7)

plt.tight_layout()
_save(fig, "02_error_metrics.png")


# ── Chart 3 : Fertilizer Match % ranked ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_PANEL)

fert_vals  = [results[n]["Fert_Match"] * 100 for n in algo_names]
order      = np.argsort(fert_vals)
names_s    = [algo_names[i] for i in order]
vals_s     = [fert_vals[i] for i in order]
colors_s   = [algo_colors[i] for i in order]

bars = ax.barh(names_s, vals_s, color=colors_s,
               edgecolor="#111", linewidth=0.5, height=0.55, zorder=3)
for bar, val in zip(bars, vals_s):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va="center", fontsize=10,
            color="white", fontweight="bold")

ax.set_title("🏆  Fertilizer Match % (Layer 2 — NPK Rule)",
             fontsize=13, fontweight="bold", color=TEXT_CLR, pad=10)
ax.set_xlabel("Match %", fontsize=10, color="#aaaacc")
ax.set_xlim(0, 115)
ax.tick_params(colors=TEXT_CLR)
ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)
ax.grid(axis="x", color=GRID_CLR, linewidth=0.5, alpha=0.7)
plt.tight_layout()
_save(fig, "03_fertilizer_match.png")


# ── Chart 4 : Full comparison heatmap ──────────────────────────────────────
metric_keys_all   = ["N_R2","K_R2","P_R2","Overall_R2","Avg_RMSE","Avg_MAE","Fert_Match"]
metric_labels_all = ["N R²","K R²","P R²","Overall R²","Avg RMSE","Avg MAE","Fert %"]

mat = np.array([[results[n][k] for n in algo_names] for k in metric_keys_all])

# Normalise each metric 0-1 for colour (RMSE/MAE inverted)
mat_norm = mat.copy()
for i, k in enumerate(metric_keys_all):
    row = mat[i]
    rng = row.max() - row.min()
    if rng == 0:
        mat_norm[i] = 0.5
    else:
        normed = (row - row.min()) / rng
        # Lower error is better → invert RMSE and MAE
        if k in ("Avg_RMSE", "Avg_MAE"):
            normed = 1 - normed
        mat_norm[i] = normed

cmap = LinearSegmentedColormap.from_list(
    "hm", ["#0d0d1a", "#4361EE", "#06D6A0", "#FFD60A"], N=256
)

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_DARK)
im = ax.imshow(mat_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(algo_names)))
ax.set_xticklabels(algo_names, rotation=20, ha="right",
                   fontsize=9, color=TEXT_CLR)
ax.set_yticks(range(len(metric_labels_all)))
ax.set_yticklabels(metric_labels_all, fontsize=10, color=TEXT_CLR)
ax.set_title("📊  All Metrics Comparison — Cleaned Dataset  (brighter = better)",
             fontsize=13, fontweight="bold", color=TEXT_CLR, pad=12)
ax.tick_params(colors=TEXT_CLR)
ax.spines[:].set_color(AXIS_CLR)

for i, k in enumerate(metric_keys_all):
    for j, name in enumerate(algo_names):
        v = mat[i, j]
        disp = f"{v*100:.1f}%" if k == "Fert_Match" else f"{v:.3f}"
        ax.text(j, i, disp, ha="center", va="center",
                fontsize=8.5, fontweight="bold",
                color="black" if mat_norm[i, j] > 0.55 else "white")

cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.ax.tick_params(colors=TEXT_CLR)
cbar.set_label("Normalised score (higher = better)", color=TEXT_CLR, fontsize=9)
plt.tight_layout()
_save(fig, "04_full_heatmap.png")


# ── Chart 5 : Actual vs Predicted NPK scatter (best model) ─────────────────
best_name = best_r2
pN = results[best_name]["pN"]
pK = results[best_name]["pK"]
pP = results[best_name]["pP"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG_DARK)
fig.suptitle(f"🔬  Actual vs Predicted NPK  [{best_name}]",
             fontsize=13, fontweight="bold", color=TEXT_CLR)

for ax, actual, predicted, label, color in [
    (axes[0], yN_te, pN, "Nitrogen",    "#4361EE"),
    (axes[1], yK_te, pK, "Potassium",   "#06D6A0"),
    (axes[2], yP_te, pP, "Phosphorous", "#F72585"),
]:
    ax.scatter(actual, predicted, alpha=0.15, s=8, color=color)
    lim = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lim, lim, "w--", linewidth=1.2, label="Perfect fit")
    r2 = r2_score(actual, predicted)
    ax.set_title(f"{label}  (R²={r2:.4f})",
                 fontsize=11, fontweight="bold", color=TEXT_CLR)
    ax.set_xlabel("Actual",    fontsize=9, color="#aaaacc")
    ax.set_ylabel("Predicted", fontsize=9, color="#aaaacc")
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors=TEXT_CLR)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)
    ax.grid(color=GRID_CLR, linewidth=0.4, alpha=0.5)

plt.tight_layout()
_save(fig, "05_actual_vs_predicted_npk.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Final summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Final Leaderboard ────────────────────────────────────────────────────")
leaderboard = sorted(results.items(),
                     key=lambda x: DIRECT_ACC[x[0]], reverse=True)

print(f"\n  {'Rank':<5} {'Algorithm':<22} "
      f"{'N R²':>8} {'K R²':>8} {'P R²':>8} "
      f"{'Overall R²':>12} {'RMSE':>8} {'MAE':>8} "
      f"{'2-Layer%':>9} {'Direct%':>9}")
print("  " + "─" * 102)

for rank, (name, res) in enumerate(leaderboard, 1):
    direct = DIRECT_ACC[name]
    print(f"  {rank:<5} {name:<22}"
          f" {res['N_R2']:>8.4f} {res['K_R2']:>8.4f} {res['P_R2']:>8.4f}"
          f" {res['Overall_R2']:>12.4f}"
          f" {res['Avg_RMSE']:>8.4f} {res['Avg_MAE']:>8.4f}"
          f" {res['Fert_Match']*100:>8.2f}%"
          f" {direct*100:>8.2f}%")

best_direct = max(DIRECT_ACC, key=DIRECT_ACC.get)
print(f"\n  🏆  Best 2-Layer  Fertilizer Match : "
      f"{max(results, key=lambda n: results[n]['Fert_Match'])}  "
      f"({max(r['Fert_Match'] for r in results.values())*100:.2f}%)")
print(f"  🏆  Best Direct Fertilizer Accuracy: {best_direct}  "
      f"({DIRECT_ACC[best_direct]*100:.2f}%)")


# ── Chart 6 : 2-layer vs Direct comparison ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG_DARK)
ax.set_facecolor(BG_PANEL)

x = np.arange(len(algo_names))
w = 0.38

bars1 = ax.bar(x - w/2,
               [results[n]["Fert_Match"]*100 for n in algo_names],
               width=w, color=[ALGO_COLORS[n] for n in algo_names],
               edgecolor="#111", linewidth=0.5, label="2-Layer (Reg→Rule)", zorder=3)
bars2 = ax.bar(x + w/2,
               [DIRECT_ACC[n]*100 for n in algo_names],
               width=w, color=[ALGO_COLORS[n] for n in algo_names],
               edgecolor="#fff", linewidth=1.0, alpha=0.65,
               label="Direct Classifier (with soil test)", zorder=3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", fontsize=7.5,
            color=TEXT_CLR, fontweight="bold")
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", fontsize=7.5,
            color=TEXT_CLR, fontweight="bold")

ax.set_title("🔄  2-Layer Pipeline vs Direct Classifier — Fertilizer Accuracy",
             fontsize=13, fontweight="bold", color=TEXT_CLR, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(algo_names, rotation=12, ha="right",
                   fontsize=9, color=TEXT_CLR)
ax.set_ylabel("Fertilizer Match %", fontsize=10, color="#aaaacc")
ax.tick_params(colors=TEXT_CLR)
ax.spines["top"].set_visible(False);  ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color(AXIS_CLR); ax.spines["left"].set_color(AXIS_CLR)
ax.grid(axis="y", color=GRID_CLR, linewidth=0.5, alpha=0.7)
ax.legend(fontsize=9, facecolor=BG_PANEL, edgecolor=AXIS_CLR,
          labelcolor=TEXT_CLR, loc="upper left")
plt.tight_layout()
_save(fig, "06_2layer_vs_direct.png")

print(f"\n  📁  Charts saved to: {os.path.abspath(RESULTS_DIR)}")
print("═" * 70 + "\n")
