# Generalisation of Dynamic Predictive Modelling

A production-oriented R&D framework for generalizing quality prediction models across manufacturing lines. This system provides end-to-end signal processing, regime discovery, and domain adaptation strategies to deploy predictive maintenance tools to new product categories with minimal retraining.

---

## Key Features & Capabilities

### Core Platform Features

* **Automated Signal Processing:** Feature extraction pipeline for 17+ high-frequency sensors (pressure, flow, vibration, temperature), including FFT spectral analysis and optional smoothing (EMA / Savitzky–Golay).
* **Dynamic Condition Monitoring:** Predicts five quality targets (Cooler, Valve, Pump, Accumulator, Stability) using a combination of classical regression and tree-based models.
* **High-Fidelity Baselines:** Uses `scipy.optimize` curve fitting as a reference against more complex machine learning models, mirroring traditional degradation laws.
* **Latent Space Monitoring:** Defines a “distance-from-good” metric in a low-dimensional embedding (UMAP) that quantifies how far a production cycle is from a typical healthy regime.
* **Modular Architecture:** Clear separation between data loading, exploration, feature engineering, and modelling so components can be reused or replaced in production.

### Advanced Capabilities

* **Unsupervised Regime Discovery:** Uses K-Means, DBSCAN, and UMAP to detect hidden operating states (for example, “good batch”, “drifted”, “early failure”) without explicit labels.
* **Domain Adaptation Engine:** Simulates production shifts between source and target domains and compares zero-shot transfer with different fine-tuning strategies.
* **Explainability:** Applies permutation importance and correlation heatmaps to rank the most influential sensors and filter out low-value features from a high-dimensional feature set.
* **Sequence Modeling:** Compares LSTM models on raw multi-sensor time-series with XGBoost on engineered features to understand the value of each approach.
* **Visual Diagnostics:** Generates latent-space projections with cluster colouring to inspect production health and regime separation.

### Enterprise-Grade Features

* **Scalability Tested:** Evaluated on 2,205 production cycles with robust handling of high-dimensional sensor arrays.
* **Dual-Mode Inference:** Supports both continuous regression (0–100% health scores) and discrete classification for stability flags.
* **Process Monitoring:** Computes drift metrics that can be used to trigger model review or retraining when the process moves away from the “good” regime.
* **Documentation:** The notebook includes a step-by-step analytical narrative, from raw signals to final models and domain-adaptation experiments.

---

## Quick Start

### Prerequisites

* Python 3.10+
* A virtual environment (recommended)

### Installation

Clone or download this repository, then set up the environment:

```bash
cd Production_Line_Modelling
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Analysis

Run the main Jupyter notebook to reproduce the analysis and models:

```bash
jupyter notebook Bosch_POC.ipynb
```

### Generate Lineage / State Visualisation

**Option 1: Latent Space Projection**

The notebook computes UMAP projections of the engineered feature space:

* Input: 400+ engineered features per cycle
* Output: 2D scatter plot showing separation between healthy and drifted batches

**Option 2: Feature Importance Heatmap**

The notebook also builds correlation and importance heatmaps:

* Output: Heatmap of the top 40 features most related to stability and other targets

---

## Repository Structure

```text
Production_Line_Modelling/
├── archive/                       # Input UCI Hydraulic dataset (PS1.txt, etc.)
├── Bosch_POC.ipynb             # Main R&D notebook
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

---

## Pipeline Overview

### Signal Processing Pipeline

* Purpose: Convert raw high-frequency sensor data into a compact feature set.
* Inputs: Raw sensor matrices (for example, PS1 at 100 Hz, FS1–FS2 at 10 Hz, TS1–TS4 at 1 Hz).
* Outputs: `X_features` feature matrix, one row per cycle.
* Main steps:

  * Smoothing with exponential moving average where appropriate.
  * FFT-based spectral features (energy, peak frequency, band powers).
  * Statistical features (moments, entropy, crest factor, autocorrelation).

### Domain Adaptation Pipeline

* Purpose: Assess and adapt model performance when moving to a new product category (Cat2).
* Inputs: Source domain (Cat1) and target domain (Cat2) samples.
* Outputs: Updated models and metrics under zero-shot and fine-tuned configurations.
* Main steps:

  * Inject synthetic domain shift (sensor scaling and offsets) to mimic changes in equipment or settings.
  * Train source-only models and evaluate zero-shot performance on Cat2.
  * Explore fine-tuning strategies, including freezing feature extractors and retraining only the prediction head.

---

## Data Dictionary

The project uses the **Condition Monitoring of Hydraulic Systems** dataset from the UCI Machine Learning Repository:

[https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)

Key signals and targets include:

* PS1–PS6: Hydraulic pressures (bar) at high sampling frequency.
* FS1–FS2: Volume flows (L/min).
* TS1–TS4: Temperatures (°C).
* VS1, CE, CP, SE, EPS1: Vibration, cooler efficiency, pump power and system efficiency indicators.
* Cooler_Condition: 3–100% efficiency (regression / ordinal target).
* Valve_Condition, Internal_Pump_Leakage, Hydraulic_Accumulator: Component health indicators.
* Stable_Flag: 0 (stable) or 1 (unstable).
* Drift_Score: Derived metric measuring distance from the “good” cluster centroid in latent space.

For a full list of columns and original descriptions, see the UCI dataset documentation and the notes inside the notebook.

---

## Adding New Models

To plug in a new model architecture:

1. Define the model in the modelling section (for example, a transformer-based sequence model).
2. Prepare data via the existing utilities (`X_features` for tabular models or sequence tensors for deep models).
3. Evaluate using the shared metric helpers for regression or classification.
4. Log results into the common results dataframe so they appear alongside current baselines.

The notebook contains examples for Random Forest, XGBoost, and LSTM models that can be followed for new additions.

---

## Technical Implementation

* Language: Python 3.10+
* Core libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `tensorflow`
* Visualisation: `matplotlib`, `seaborn`, `umap-learn`
* Data formats: Text / CSV for sensor and label files; in-memory arrays and dataframes for processing
* Data source: Non-personal industrial sensor data (UCI hydraulic condition monitoring dataset)

---

## Next Steps

Examples of natural extensions:

* Wrap feature extraction and inference into an API service (for example, using FastAPI).
* Add drift thresholds around the latent-space distance metric and expose alerts.
* Experiment with multi-site or federated learning setups for multiple plants.
* Integrate with real-time data sources or PLCs for online monitoring.
* Build a dashboard (for example, with Streamlit) for operators to view current health, drift trends, and predictions.

---

## Success Criteria Achieved

This project implements and documents the main requirements of the Bosch “Generalisation of Dynamic Predictive Modelling” problem statement.

### Core Requirements

* Existing approach understanding: Recreates traditional curve-fitting baselines and shows that simple regression can reach very high R² (around 0.99) on Cooler Condition, providing a reference for more complex models.
* Unsupervised clustering: Uses K-Means and DBSCAN to separate normal and drifted regimes and to identify early-failure-like clusters.
* Feature optimisation: Builds a large feature set per sensor (including spectral and statistical features) and uses permutation importance to focus on the most robust inputs.
* Retraining for new domains: Simulates a realistic product category shift and demonstrates that zero-shot transfer and head-only fine-tuning can maintain high AUC (around 0.95) on the target domain.
* Technical deliverables: Provides a working notebook showing signal processing, pattern recognition, and domain-adaptation workflows.
* Repository and documentation: Includes a clear README, folder structure, and narrative description of the pipeline.

### Additional Enhancements

* Model benchmarking: Direct comparison of Random Forest, XGBoost, and LSTM shows that feature-based tree models (for example, XGBoost with RMSE around 0.12 for stability) outperform the raw-sequence LSTM baseline (RMSE around 0.24) for this dataset.
* Latent-space health metric: Defines a quantitative distance-from-good score based on UMAP embeddings that can be used for monitoring.
* Mapping to operational grades: Includes “nearest-grade” mapping from regression outputs back to physically meaningful discrete levels (for example, 100%, 90%, etc.).
* Modular design: Keeps feature extraction, model training, and evaluation loosely coupled to support different deployment patterns.
* Broader testing: Evaluates models across multiple failure modes and sensors, not only on a single target.
* Visual analysis: Provides plots and projections that make it easier to understand how regimes and drifts appear in the data.

### Example Metrics

* Predictive accuracy: R² above 0.99 for Cooler Condition; RMSE below 0.13 for stability prediction with XGBoost on engineered features.
* Domain transfer: AUC around 0.95 when transferring from Category 1 (source) to Category 2 (target).
* Cluster separation: Latent-space clustering shows well-separated operating regimes, with supporting statistics reported in the notebook.

---

## License

This project is a proof of concept for educational and demonstration purposes.
