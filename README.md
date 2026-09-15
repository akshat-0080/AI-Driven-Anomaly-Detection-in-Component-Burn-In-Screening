# AI-Driven Anomaly Detection in Component Burn-In Screening

This repository contains Python scripts to detect defective microchips during burn-in screening tests using machine learning.

During burn-in testing, semiconductor components are stressed under high temperature and voltage for set periods: 0 hours, 24 hours, 96 hours, and 168 hours. The goal is to catch early-life failures before components ship to customers. This project processes electrical measurement logs recorded across those timepoints and flags anomalous parts.

---

## What the Code Does

Instead of relying on fixed manual thresholds, this project processes time-series test data to identify abnormal degradation patterns automatically.

### 1. Data Ingestion & Preprocessing (`src/data_preprocessing.py`)
* Reads tabular semiconductor test data containing physical metrics like $I_{\text{ddq}}$ (quiescent supply current) and propagation delay across test duration milestones ($0\text{h}$, $24\text{h}$, $96\text{h}$, $168\text{h}$).
* Cleans missing values, normalizes metrics, and organizes readings by component ID.

### 2. Feature Extraction (`src/feature_engineering.py`)
* Computes degradation rates across consecutive time steps ($\Delta 0\rightarrow24\text{h}$, $\Delta 24\rightarrow96\text{h}$, $\Delta 96\rightarrow168\text{h}$).
* Measures total parameter drift, volatility, and correlation across metrics (such as checking if timing delay increases alongside current leakage).

### 3. Anomaly Detection (`src/model_training.py`)
* Uses unsupervised models (Isolation Forest, One-Class SVM) and supervised models (Random Forest, XGBoost) to score components based on how their degradation behavior deviates from normal devices.
* Flags components as healthy baseline chips, timing outliers, leakage anomalies, or sensor glitches.

### 4. Evaluation (`src/evaluation.py`)
* Calculates performance metrics including Precision, Recall, F1-Score, and ROC-AUC.
* Generates plots showing parameter drift over time to help engineers inspect flagged components.

---

## Repository Structure

```text
AI-Driven-Anomaly-Detection-in-Component-Burn-In-Screening/
│
├── data/
│   ├── raw/                  # Input datasets with burn-in measurements (0h - 168h)
│   └── processed/            # Cleaned data with engineered drift features
│
├── src/
│   ├── data_preprocessing.py # Script for cleaning and scaling metrics
│   ├── feature_engineering.py# Script for computing delta parameters across timepoints
│   ├── model_training.py     # Script to train anomaly detection and classification models
│   └── evaluation.py         # Script to generate metrics, reports, and drift charts
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive data analysis
│
├── models/                   # Saved trained model files (.pkl)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 2. Real-World Realism Evaluation

**Verdict:** No, they do not replicate real-world semiconductor data naturally. The files are synthetic benchmark datasets. While they model standard burn-in timepoints ($0\text{h}$, $24\text{h}$, $96\text{h}$, $168\text{h}$), they exhibit specific synthetic artifacts that differ from real-world CMOS semiconductor physics:

### Linear vs. Sub-Linear Aging Kinetics
* **Dataset Behavior:** Normal baseline degradation grows at a strict linear rate ($\approx 0.0018\,\mu\text{A/hr}$ constantly across $0\rightarrow24\text{h}$, $24\rightarrow96\text{h}$, and $96\rightarrow168\text{h}$).
* **Real-World Physics:** Physical silicon aging (due to NBTI, HCI, or TDDB) follows power-law kinetics ($\Delta P \propto t^n$, where $n \approx 0.16 - 0.25$). Degradation is steeper during initial burn-in ($0\rightarrow24\text{h}$) and saturates over time.

### Synthetic Sensor Glitch Dynamics
* **Dataset Behavior:** Sensor_Glitch parts exhibit isolated single-point spikes (e.g., $I_{\text{ddq}}$ jumping from $11.8\,\mu\text{A}$ to $23.9\,\mu\text{A}$ at $24\text{h}$ and returning to $12.1\,\mu\text{A}$ at $96\text{h}$).
* **Real-World Physics:** Sudden transient spikes in test data usually stem from test socket contact resistance or ATE noise rather than underlying semiconductor physical defects.

### Decoupled Parameter Drift
* **Dataset Behavior:** Outlier classes like Timing_Outlier show severe propagation delay surges (up to $24.99\,\text{ns}$) while $I_{\text{ddq}}$ remains completely nominal.
* **Real-World Physics:** Transistor speed and leakage are intrinsically coupled through threshold voltage ($V_{\text{th}}$). A major degradation in propagation delay almost always manifests alongside quiescent current shifts in physical ICs.

---

## Getting Started

### Prerequisites
Python 3.8 or higher.

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/akshat-0080/AI-Driven-Anomaly-Detection-in-Component-Burn-In-Screening.git](https://github.com/akshat-0080/AI-Driven-Anomaly-Detection-in-Component-Burn-In-Screening.git)
   cd AI-Driven-Anomaly-Detection-in-Component-Burn-In-Screening
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

1. Preprocess data and engineer features:
   ```bash
   python src/data_preprocessing.py
   python src/feature_engineering.py
   ```

2. Train models:
   ```bash
   python src/model_training.py
   ```

3. Generate evaluation metrics:
   ```bash
   python src/evaluation.py
   ```
