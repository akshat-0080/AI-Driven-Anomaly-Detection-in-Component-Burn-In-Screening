import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import confusion_matrix

# =====================================================================
# 1. LOAD DATA & INITIAL CONFIGURATION
# =====================================================================
FILE_PATH = '/content/significant_changes_training_data.csv'  # Replace with your actual filename if different
df = pd.read_csv(FILE_PATH)

# Target encoding: Faulty / Anomaly = 1, Normal = 0
df['Target'] = (df['Ground_Truth'] != 'Normal').astype(int)

# Train (80%) / Test (20%) split stratified by ground truth
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df['Target']
)

# =====================================================================
# 2. FEATURE ENGINEERING ENGINE (0h to 24h Early Screening Window)
# =====================================================================
def generate_screening_features(data_frame: pd.DataFrame) -> pd.DataFrame:
    df_feat = data_frame.copy()
    params = ['Iddq_uA', 'Leakage_nA', 'PropDelay_ns']
    
    # Temporal Drift & Slope features from early screening (0h -> 24h)
    for p in params:
        p0, p24 = f'{p}_0h', f'{p}_24h'
        df_feat[f'{p}_drift_24h'] = df_feat[p24] - df_feat[p0]
        df_feat[f'{p}_rel_drift'] = df_feat[f'{p}_drift_24h'] / (np.abs(df_feat[p0]) + 1e-6)
        df_feat[f'{p}_slope_24h'] = df_feat[f'{p}_drift_24h'] / 24.0
        df_feat[f'{p}_linear_proj_168h'] = df_feat[p0] + (df_feat[f'{p}_slope_24h'] * 168.0)
    
    # Cross-parameter interactions (decoupling indicators)
    df_feat['leak_per_iddq_24h'] = df_feat['Leakage_nA_24h'] / (df_feat['Iddq_uA_24h'] + 1e-5)
    df_feat['delay_per_iddq_24h'] = df_feat['PropDelay_ns_24h'] / (df_feat['Iddq_uA_24h'] + 1e-5)

    # Module A: Lot-aware Modified Z-Scores (MAD strategy)
    for p in params:
        for suffix in ['0h', '24h', 'drift_24h']:
            col = f'{p}_{suffix}'
            def calc_mod_z(s):
                med = s.median()
                mad = np.median(np.abs(s - med))
                return 0.6745 * (s - med) / (mad + 1e-8)
            
            df_feat[f'{col}_mod_z'] = df_feat.groupby('Lot_ID')[col].transform(calc_mod_z)
            
    return df_feat

train_feat = generate_screening_features(train_df)
test_feat = generate_screening_features(test_df)

# Feature matrix setup
mod_z_cols = [c for c in train_feat.columns if c.endswith('_mod_z')]
feature_cols = [
    'Iddq_uA_0h', 'Iddq_uA_24h', 'Iddq_uA_drift_24h', 'Iddq_uA_rel_drift', 'Iddq_uA_slope_24h', 'Iddq_uA_linear_proj_168h',
    'Leakage_nA_0h', 'Leakage_nA_24h', 'Leakage_nA_drift_24h', 'Leakage_nA_rel_drift', 'Leakage_nA_slope_24h', 'Leakage_nA_linear_proj_168h',
    'PropDelay_ns_0h', 'PropDelay_ns_24h', 'PropDelay_ns_drift_24h', 'PropDelay_ns_rel_drift', 'PropDelay_ns_slope_24h', 'PropDelay_ns_linear_proj_168h',
    'leak_per_iddq_24h', 'delay_per_iddq_24h'
] + mod_z_cols

# =====================================================================
# 3. MODULE A: DYNAMIC LOT-LEVEL OUTLIER SCREENING
# =====================================================================
Z_THRESHOLD = 2.5
test_feat['module_a_reject'] = (test_feat[mod_z_cols].abs() > Z_THRESHOLD).any(axis=1)

# =====================================================================
# 4. MODULE B: TIME-SERIES REGRESSION & GUARDED PREDICTIVE SCREENING
# =====================================================================
# Train Gradient Boosting Classifier tuned for hyper-sensitive detection
clf = HistGradientBoostingClassifier(random_state=42, max_iter=200)
clf.fit(train_feat[feature_cols], train_feat['Target'])

# Predict anomaly probabilities on test set
test_probs = clf.predict_proba(test_feat[feature_cols])[:, 1]

# Train 168h Parameter Regressors for Guarded Safety Slope Validation
reg_iddq = HistGradientBoostingRegressor(random_state=42).fit(train_feat[feature_cols], train_feat['Iddq_uA_168h'])
reg_leak = HistGradientBoostingRegressor(random_state=42).fit(train_feat[feature_cols], train_feat['Leakage_nA_168h'])
reg_delay = HistGradientBoostingRegressor(random_state=42).fit(train_feat[feature_cols], train_feat['PropDelay_ns_168h'])

# Forecast 168h end-of-life state
test_feat['pred_iddq_168h'] = reg_iddq.predict(test_feat[feature_cols])
test_feat['pred_leak_168h'] = reg_leak.predict(test_feat[feature_cols])
test_feat['pred_delay_168h'] = reg_delay.predict(test_feat[feature_cols])

# Derive safety slope limits from 99th percentile of known normal training parts
normal_train = train_feat[train_feat['Target'] == 0]
limit_iddq_slope = (normal_train['Iddq_uA_168h'] - normal_train['Iddq_uA_0h']).quantile(0.99) / 168.0
limit_leak_slope = (normal_train['Leakage_nA_168h'] - normal_train['Leakage_nA_0h']).quantile(0.99) / 168.0
limit_delay_slope = (normal_train['PropDelay_ns_168h'] - normal_train['PropDelay_ns_0h']).quantile(0.99) / 168.0

# Calculate predicted slopes on test set
test_feat['pred_iddq_slope'] = (test_feat['pred_iddq_168h'] - test_feat['Iddq_uA_0h']) / 168.0
test_feat['pred_leak_slope'] = (test_feat['pred_leak_168h'] - test_feat['Leakage_nA_0h']) / 168.0
test_feat['pred_delay_slope'] = (test_feat['pred_delay_168h'] - test_feat['PropDelay_ns_0h']) / 168.0

slope_violation = (
    (test_feat['pred_iddq_slope'] > limit_iddq_slope) |
    (test_feat['pred_leak_slope'] > limit_leak_slope) |
    (test_feat['pred_delay_slope'] > limit_delay_slope)
)

# Ultra-pessimistic decision threshold (0.05) to prioritize Recall over Precision
GUARDED_PROB_THRESHOLD = 0.05
test_feat['module_b_reject'] = (test_probs > GUARDED_PROB_THRESHOLD) | slope_violation

# Final Combined Screening Decision
test_feat['Final_Decision'] = (test_feat['module_a_reject'] | test_feat['module_b_reject']).astype(int)

# =====================================================================
# 5. METRICS REPORTING
# =====================================================================
y_true = test_feat['Target'].values
y_pred = test_feat['Final_Decision'].values

# Confusion matrix breakdown
# TN: Working kept, FP: Working rejected, FN: Faulty missed, TP: Faulty caught
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print("=" * 60)
print(f"       ESS SCREENING MODEL TEST RESULTS (Total Tested: {len(test_feat)})")
print("=" * 60)
print(f"Total Faulty Components in Test Set   : {y_true.sum()}")
print(f"Total Working Components in Test Set  : {(y_true == 0).sum()}")
print("-" * 60)
print(f"Faulty components deemed fine (FN)   : {fn}")
print(f"Working components rejected (FP)      : {fp}")
print("-" * 60)
print(f"Faulty Capture Rate (Recall)          : {(tp / (tp + fn)) * 100:.2f}%")
print(f"Yield Loss Rate (False Positives)     : {(fp / (tn + fp)) * 100:.2f}%")
print("=" * 60)
