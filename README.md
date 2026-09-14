# Module A: Multi-Stage Electrical Burn-In Outlier Detection Pipeline

**Aim:** Detect subtle anomalous components operating within standard static specification limits via a progressive 3-stage statistical and machine learning pipeline.

---

## 1. Architecture & Strategy

Detecting near-spec electrical failures during Environmental Stress Screening (ESS) requires balancing fast early rejection with dynamic multivariate modeling. Module A cascades data through a three-stage screening funnel to isolate defective units while minimizing computational overhead on normal components:

| Stage | Focus | Primary Mechanism | Target Anomalies |
| :--- | :--- | :--- | :--- |
| **Stage 1** | $t=0\text{h}$ Baseline Screening | Univariate Modified Z-Scores (MAD) | Gross static leakage & baseline parameter shifts |
| **Stage 2** | Temporal Drift & Acceleration | Multi-point time series screening ($24\text{h}, 96\text{h}, 168\text{h}$) | Parametric drift & second-order burn-in acceleration |
| **Stage 3** | Multivariate ML Engine | Robust-scaled Random Forest with Guarded Thresholding | Sub-threshold non-linear interactions & residual anomalies |

---

## 2. Advanced Feature Engineering

To amplify minor parametric anomalies before model training, raw electrical parameters (`iddq_ua`, `leakage_na`, `propdelay_ns`) are transformed across temporal and interaction dimensions:

* **Absolute Drift:** Measures raw parameter degradation over time intervals $t \in \{24\text{h}, 96\text{h}, 168\text{h}\}$:
  $$\Delta X_t = X_t - X_{0\text{h}}$$

* **Relative Drift:** Amplifies dynamic shifts relative to initial baseline:
  $$\Delta X_{\text{rel}, t} = \frac{\Delta X_t}{|X_{0\text{h}}| + 10^{-5}}$$

* **Burn-In Acceleration:** Captures second-order derivatives between screening checkpoints ($24\text{h}\to 96\text{h}$ and $96\text{h}\to 168\text{h}$) to isolate volatile degradation paths:
  $$a_{96\_24} = \Delta X_{96\text{h}} - \Delta X_{24\text{h}}, \quad a_{168\_96} = \Delta X_{168\text{h}} - \Delta X_{96\text{h}}$$

* **Cross-Parameter Ratios:** Identifies functional decoupling between static current modes and timing performance.

---

## 3. Multi-Stage Detection Methodology

### Stage 1: Baseline Screening ($t=0\text{h}$)
To prevent extreme outliers from skewing variance, parameter standardization uses Median Absolute Deviation (MAD) calculated exclusively on the training distribution:

$$\tilde{Z} = 0.6745 \cdot \frac{|X - \text{Median}|}{\text{MAD}}$$

* **Static Leakage Outlier:** Triggered if $\tilde{Z}_{\text{leakage}} \ge 2.5$.
* **General Parameter Outlier**
* **Coincidence Screening:** Outlier if $\ge 2$ parameters exhibit $\tilde{Z} \ge 2.3$.
* **Watchlist Routing:** Borderline units ($1.8 \le \max(\tilde{Z}) < 3.2$) are placed on a Watchlist for prioritized Stage 2 drift evaluation.

### Stage 2: Drift & Acceleration Screening ($t=24\text{h}, 96\text{h}, 168\text{h}$)
Evaluates Modified Z-scores across all absolute drift and acceleration vectors ($\tilde{Z}_{\text{accel}}$):

* **Extreme Drift/Acceleration** 
* **Watchlist Escalation:** Units flagged on the Stage 1 Watchlist are rejected if drift exceeds a lowered threshold ($\max(\tilde{Z}_{\text{drift}}) \ge 2.1$).

### Stage 3: Multivariate ML Engine & Guarded Floor Calibration
Remaining components are evaluated by a 350-tree Random Forest classifier fitted with balanced subsampling. Inputs are scaled using `RobustScaler` (median/IQR) to restrict outlier influence on feature scaling.

To strictly minimize **False Negatives (FN)** without allowing the **False Positive (FP)** scrap rate to explode:

1. **Uncaught Training Anomaly Identification:** The pipeline tracks anomalies ($y=1$) missed by Stage 1 and Stage 2 screening on the training set.
2. **Adaptive Threshold Floor:** The decision probability threshold is dynamically set to target residual training defects, guarded by a minimum floor of $0.18$:
   $$\theta_{\text{ML}} = \max\left(0.18,\, 0.98 \cdot \min_{i \in \text{Uncaught}} P(y_i = 1)\right)$$

---

## 4. False Negative Mitigation & Scrap Rate Control

* **False Negative (FN) Minimization:** Standard fixed thresholds often miss subtle multi-dimensional anomalies that fall just under statistical limits. Stage 3 solves this by dynamically lowering the Random Forest decision threshold ($\theta_{\text{ML}}$) to capture the lowest probability assigned to an uncaught training anomaly (guarded at $0.18$). This prevents leakage of dangerous borderline components into production.
* **False Positive (FP) / Scrap Rate Control:** Hard univariate limits (e.g., fixed standard deviation limits) frequently scrap healthy, high-performing components due to natural process variance. By employing **MAD-based robust statistics** instead of standard deviation, **RobustScaler** for non-distorted feature spaces, and **multi-parameter coincidence logic**, normal distribution tails are protected—keeping false scrap rates low while maintaining maximum defect sensitivity.
