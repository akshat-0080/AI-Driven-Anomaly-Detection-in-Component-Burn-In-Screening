import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# =====================================================================
# 1. LOAD DATASET & ADVANCED FEATURE ENGINEERING
# =====================================================================
file_path = "ess_test_data_electrical_v2.csv"
df = pd.read_csv(file_path)

# Standardize column headers
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 1a. Absolute & Relative Drift Features (Captures Subtle Parametric Shifts)
time_points = [24, 96, 168]
drift_cols = []
rel_drift_cols = []
for t in time_points:
    for param in ["iddq_ua", "leakage_na", "propdelay_ns"]:
        d_col = f"{param}_drift_{t}h"
        r_col = f"{param}_rel_drift_{t}h"
        
        df[d_col] = df[f"{param}_{t}h"] - df[f"{param}_0h"]
        # Relative percentage change to amplify minor dynamic shifts
        df[r_col] = df[d_col] / (df[f"{param}_0h"].abs() + 1e-5)
        
        drift_cols.append(d_col)
        rel_drift_cols.append(r_col)

# 1b. Acceleration Features (Second-order burn-in derivative for Dynamic_Outlier)
accel_cols = []
for param in ["iddq_ua", "leakage_na", "propdelay_ns"]:
    col1 = f"{param}_accel_96_24"
    col2 = f"{param}_accel_168_96"
    df[col1] = df[f"{param}_drift_96h"] - df[f"{param}_drift_24h"]
    df[col2] = df[f"{param}_drift_168h"] - df[f"{param}_drift_96h"]
    accel_cols.extend([col1, col2])

# 1c. Cross-Parameter Interactions
df["leakage_iddq_ratio_0h"] = df["leakage_na_0h"] / (df["iddq_ua_0h"] + 1e-5)
df["prop_iddq_ratio_0h"] = df["propdelay_ns_0h"] / (df["iddq_ua_0h"] + 1e-5)

# Stratified 90% Train / 10% Test Split
train_df, test_df = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["ground_truth"]
)
train_df = train_df.copy()
test_df = test_df.copy()

print(f"Dataset Split: {len(train_df)} Train Samples (90%), {len(test_df)} Test Samples (10%)\n")

cols_0h = ["iddq_ua_0h", "leakage_na_0h", "propdelay_ns_0h"]
all_stat_cols = cols_0h + drift_cols + accel_cols

ignore_cols = ["component_id", "lot_id", "ground_truth"]
ml_feature_cols = [c for c in train_df.columns if c not in ignore_cols]

# =====================================================================
# 2. STATISTICAL UTILITIES
# =====================================================================
def fit_mad_statistics(train_data, columns):
    """Computes Median and MAD statistics exclusively on Training set."""
    stats = {}
    for col in columns:
        median = train_data[col].median()
        mad = (train_data[col] - median).abs().median()
        if mad == 0:
            mad = 1e-6
        stats[col] = (median, mad)
    return stats

def compute_zscore_matrix(target_data, stats, columns):
    """Computes Modified Z-score matrix for target parameters."""
    z_matrix = pd.DataFrame(index=target_data.index)
    for col in columns:
        median, mad = stats[col]
        z_matrix[col] = 0.6745 * (target_data[col] - median).abs() / mad
    return z_matrix

stats_all = fit_mad_statistics(train_df, all_stat_cols)

# =====================================================================
# 3. STAGE 1 & 2: REFINED STATISTICAL SCREENING
# =====================================================================
def run_statistical_screening(dataset):
    z_0h = compute_zscore_matrix(dataset, stats_all, cols_0h)
    
    # Stage 1: Targeted threshold for baseline leakage (Z >= 2.5) isolates static Leakage_Only_Outlier
    leakage_s1 = z_0h["leakage_na_0h"] >= 2.5
    other_s1 = z_0h[["iddq_ua_0h", "propdelay_ns_0h"]].max(axis=1) >= 3.2
    s1_coincidence = (z_0h >= 2.3).sum(axis=1) >= 2
    s1_watchlist = (z_0h.max(axis=1) >= 1.8) & ~leakage_s1 & ~other_s1 & ~s1_coincidence
    
    stage1_flag = np.select(
        [leakage_s1 | other_s1 | s1_coincidence, s1_watchlist],
        ["Stage1_Outlier", "Watchlist"],
        default="Pass"
    )
    
    # Stage 2: Drift & Acceleration Screening
    z_drift = compute_zscore_matrix(dataset, stats_all, drift_cols)
    z_accel = compute_zscore_matrix(dataset, stats_all, accel_cols)
    
    s2_drift_extreme = z_drift.max(axis=1) >= 3.0
    s2_accel_extreme = z_accel.max(axis=1) >= 3.0
    s2_watchlist_drift = (stage1_flag == "Watchlist") & (z_drift.max(axis=1) >= 2.1)
    
    stage2_flag = np.where(
        s2_drift_extreme | s2_accel_extreme | s2_watchlist_drift,
        "Stage2_Drift_Outlier",
        "Pass"
    )
    return stage1_flag, stage2_flag

train_s1_flag, train_s2_flag = run_statistical_screening(train_df)
test_df["stage1_flag"], test_df["stage2_flag"] = run_statistical_screening(test_df)

# =====================================================================
# 4. STAGE 3: MACHINE LEARNING ENGINE WITH ROBUST SCALER & GUARDED FLOOR
# =====================================================================
# RobustScaler uses median and IQR, preventing extreme outliers from compressing normal distribution
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_df[ml_feature_cols])
X_test_scaled = scaler.transform(test_df[ml_feature_cols])

y_train_binary = (train_df["ground_truth"] != "Normal").astype(int)
y_test_binary = (test_df["ground_truth"] != "Normal").astype(int)

rf_model = RandomForestClassifier(
    n_estimators=350,
    max_depth=15,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42
)
rf_model.fit(X_train_scaled, y_train_binary)

# Evaluate uncaught training anomalies to set minimum sensitivity threshold
train_caught_by_s1_s2 = (train_s1_flag == "Stage1_Outlier") | (train_s2_flag == "Stage2_Drift_Outlier")
train_probs = rf_model.predict_proba(X_train_scaled)[:, 1]
uncaught_train_mask = (y_train_binary == 1) & (~train_caught_by_s1_s2)

if uncaught_train_mask.sum() > 0:
    min_uncaught_prob = float(train_probs[uncaught_train_mask].min())
    # Guard floor at 0.18 to prevent threshold collapse while retaining full sensitivity
    ml_threshold = max(0.18, min_uncaught_prob * 0.98)
else:
    ml_threshold = 0.30

test_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
test_df["stage3_flag"] = np.where(test_probs >= ml_threshold, "Stage3_ML_Outlier", "Pass")

# =====================================================================
# 5. PIPELINE AGGREGATION & EVALUATION
# =====================================================================
test_df["predicted_anomaly"] = (
    (test_df["stage1_flag"] == "Stage1_Outlier") |
    (test_df["stage2_flag"] == "Stage2_Drift_Outlier") |
    (test_df["stage3_flag"] == "Stage3_ML_Outlier")
)
test_df["actual_anomaly"] = y_test_binary == 1

acc = accuracy_score(test_df["actual_anomaly"], test_df["predicted_anomaly"])
prec = precision_score(test_df["actual_anomaly"], test_df["predicted_anomaly"], zero_division=0)
rec = recall_score(test_df["actual_anomaly"], test_df["predicted_anomaly"], zero_division=0)
f1 = f1_score(test_df["actual_anomaly"], test_df["predicted_anomaly"], zero_division=0)

normal_mask = test_df["ground_truth"] == "Normal"
total_normals = normal_mask.sum()
fp_normals = (test_df["predicted_anomaly"] & normal_mask).sum()
scrap_rate = (fp_normals / total_normals) * 100

print(f"Calibrated ML Decision Threshold: {ml_threshold:.4f}")
print(f"Normal Component False Scrap Rate: {scrap_rate:.2f}% ({fp_normals}/{total_normals} good parts rejected)\n")

print("=== PIPELINE EVALUATION METRICS (10% TEST SET) ===")
print(f"Accuracy:  {acc:.2%}")
print(f"Precision: {prec:.2%}")
print(f"Recall:    {rec:.2%}")
print(f"F1-Score:  {f1:.2%}")
print("-" * 50)

print("\n=== CONFUSION MATRIX BY GROUND TRUTH CLASS ===")
print(pd.crosstab(
    test_df["ground_truth"],
    test_df["predicted_anomaly"],
    rownames=["Ground Truth"],
    colnames=["Predicted Anomaly"]
))
