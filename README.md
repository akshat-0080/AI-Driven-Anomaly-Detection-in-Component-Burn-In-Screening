# AI-Driven-Anomaly-Detection-in-Component-Burn-In-Screening
PROBLEM STATEMENT - Background In high-reliability sectors (like space) electronic components undergo rigorous environmental stress screening (ESS), including Burn-In testing (operating components at elevated temperatures, e.g., 125Â°C for extended periods).

Traditional screening relies on static parametric pass/fail limits. However, 'latent defects' components that pass the absolute limits but exhibit subtle, anomalous drift over time often escape into final payloads, leading to catastrophic field failures.

Description Development of a predictive machine learning model that analyzes time-series parametric data (e.g., standby current Iddq, leakage currents, or propagation delays measured at intervals like 0h, 24h, 96h, and 168h to detect anomalous components.

Expected Solution Module A: The outlier detection system Static limits catch obvious failures. Participants need to develop a 'Dynamic' outlier detection system. If a lot has an average leakage current of 10ÂµA, a part showing 45 ÂµA is a massive anomaly, even if the absolute datasheet maximum limit is 50 ÂµA.

Module B: Time-Series Drift Predictor Build a predictive regression model that takes Value_0h and Value_24h as inputs and forecasts Value_168h. If the predicted 168h drift rate exceeds a calculated safety slope, the system flags the component for early rejection.

Evaluation Metrics:

• Anomaly Detection Score: a False Negative (missing a defective part) is catastrophic, penalizing teams that let bad parts escape.
• Drift Prediction Accuracy : The mean absolute error between the predicted Value_168h and the actual hidden ground-truth values.
• Explainability : Can the model justify its classification to a QA inspector, or is it a complete black box?


##                                                                      MODULE A

**Aim:** Identify outlier components operating within standard static limits (e.g., $< 1\text{A}$) using a multi-stage statistical and machine learning pipeline.

---

### 1. Overview & Strategy

Detecting subtle component anomalies requires balancing fast early-rejection with deep multivariate analysis. Module A uses a progressive detection hierarchy:

1. **Stage 1 ($t = 0\text{h}$ Baseline Screening):** Fast univariate statistical filtering to catch immediate extreme outliers without heavy computation.
2. **Stage 2 ($t = 24\text{h}$ Temporal Drift Analysis):** Tracking borderline components over time to identify temporal degradation.
3. **Stage 3 (Multivariate Isolation Forest):** Capturing complex inter-dependencies across multiple current types.
    We first bring down the size of our data set using a quick stage 1 and implement a much tighter but computation heavy bound in the following stages.

---

### 2. Methodology & Mathematical Formulation

#### A. Stage 1: Initial Univariate Filtering ($t = 0\text{h}$)
At $t = 0\text{h}$, temporal correlations are zero. Assuming homogeneous current channels, we evaluate components independently using standardized scores:

$$Z = \frac{X - \mu}{\sigma} \quad \text{or} \quad \tilde{Z} = \frac{0.6745 \cdot (X - \text{Median})}{\text{MAD}}$$

* **Extreme Outliers ($|Z| \ge 3.5 - 4.0$):** Flagged for early rejection.
* **Borderline Candidates ($2.0 \le |Z| < 3.5$):** Stored in a watchlist for Stage 2 temporal validation.

> **Note on Thresholding:** To prevent information loss prior to tree-based models, hard clipping at $|Z| = 3.0$ is avoided. Maintaining tail variance improves subsequent Isolation Forest tree-splitting efficiency.

#### B. Stage 2: Temporal Drift Screening ($t = 24\text{h}$)
Borderline components from $t = 0\text{h}$ are re-evaluated at $t = 24\text{h}$. If a component drifts such that its $t = 24\text{h}$ score satisfies $|Z_{24}| > 3.0$, it is categorized as a temporal degradation outlier.

#### C. Stage 3: Multivariate Isolation Forest
When accounting for multiple current modes (e.g., static, dynamic, idle rail currents), univariate independence no longer holds. Stage 3 feeds unclipped multi-channel current data into an **Isolation Forest (IF)** model to isolate anomalous components in high-dimensional feature space.

---

### 3. References & Further Reading

* **Z-Score Research Paper:** [ArXiv: Outlier Detection Analysis](https://arxiv.org/pdf/2604.08581)
* **Isolation Forest Methodology:** [Module Documentation / Reference](https://drive.google.com/file/d/1wAi7Tzem09ohslAgTRyOg0EfL4I4i57K/view?usp=sharing) 
