# Dynamic Predictive Modelling for Hydraulic Systems

This repository contains an end-to-end R&D project focused on **"Generalisation of Dynamic Predictive Modelling Across Production Lines."** The project builds predictive models for component health and system stability using multi-sensor hydraulic time-series data. A key focus of the work is studying how well these models generalize across different operating conditions and product categories, simulating real-world industrial domain shifts.

## 1. Dataset

The project utilizes the public **Condition Monitoring of Hydraulic Systems** dataset as a proxy for industrial production line data.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- **Data Structure:** Cycle-based time-series from multiple sensors, including:
  - **Pressure:** PS1–PS6
  - **Flow:** FS1–FS2
  - **Temperature:** TS1–TS4
  - **Vibration:** VS1
  - **Efficiency/Power:** CE, CP, SE, EPS1
- **Target Variables:** Five labels per cycle (derived from `profile.txt`):
  - `Cooler_Condition`
  - `Valve_Condition`
  - `Internal_Pump_Leakage`
  - `Hydraulic_Accumulator`
  - `Stable_Flag` (Binary: 0/1)

## 2. Methodology

### 2.1 Data Preparation
- Loaded raw sensor matrices and aligned cycles.
- Joined sensor data with the five target labels into a unified dataframe (`df_final`).

### 2.2 Feature Engineering
Comprehensive feature extraction was performed for every sensor and cycle, generating a wide tabular feature matrix:
- **Time-domain:** Mean, Std Dev, RMS, Crest Factor, Quantiles, Lag-1 Autocorrelation, Entropy, Skewness, Kurtosis.
- **Frequency-domain:** Spectral Energy, Spectral Centroid, Bandwidth, Band Powers.

### 2.3 Domain Shift Simulation
To simulate the challenge of deploying models to new production lines:
- Created a synthetic `Product_Category` split (Cat1 / Cat2).
- Applied controlled shifts to a subset of features in Cat2 to mimic "target domain" variance.

### 2.4 Modelling Strategy
- **Baselines:** Linear regression, polynomial regression, and simple one-feature curve fits.
- **Feature-based ML:** `RandomForestRegressor` and `XGBRegressor` trained on engineered features.
- **Sequence Models:** `LSTM` networks trained on raw resampled sequences (Dimensions: *n_cycles × timesteps × n_channels*).

### 2.5 Unsupervised Analysis
- **Clustering:** KMeans and DBSCAN applied to the feature space to identify operating regimes.
- **Explainability:** Correlation analysis and permutation importance to identify critical sensors.

## 3. Key Results & Insights

### 3.1 Predictive Performance
* **Cooler Condition:** Highly predictable. One-feature curve fits achieved **R² ≈ 0.998** (RMSE ≈ 1.77).
* **Hydraulic Accumulator:** Non-linear but learnable. One-feature curve fits achieved **R² ≈ 0.56**.
* **Valve & Pump Leakage:** Moderate signal strength. Curve fits achieved **R² ≈ 0.50** and **0.76** respectively.
* **System Stability (`Stable_Flag`):**
    * **Challenge:** Simple baselines struggled (R² ≈ 0.26).
    * **Solution:** Feature-based tree models outperformed deep learning for this specific task.
        * **XGBoost:** RMSE ≈ 0.128, MAE ≈ 0.040
        * **Random Forest:** RMSE ≈ 0.146, MAE ≈ 0.052
        * **LSTM:** RMSE ≈ 0.243 (Underperformed, suggesting engineered features captured the necessary temporal structure).

### 3.2 Feature Importance & Regimes
* **Critical Features:** Permutation importance highlighted higher-order statistics (Skew, Kurtosis, Crest Factor) in **PS2** (Pressure) and **FS1** (Flow), indicating that stability is defined by *waveform shape* changes rather than simple mean values.
* **Clustering:** KMeans identified a distinct "Good" cluster (~1959 cycles) vs. a "Noisy" cluster (~246 cycles).
* **Hidden States:** Analysis revealed an "incipient degradation" regime characterized by lower cooler efficiency (~41%) and mild valve degradation (~91%), allowing for early failure detection.

### 3.3 Domain Adaptation (Cat1 → Cat2)
* **Zero-Shot Transfer:** Models trained on Source (Cat1) and tested on Target (Cat2) achieved **AUC ≈ 0.955** for stability prediction.
* **Fine-Tuning:** Freezing the feature extractor and retraining the head yielded similar performance to full fine-tuning.
* **Insight:** Naïve re-weighting strategies harmed performance (AUC dropped to ≈ 0.51), suggesting robust feature representation is more effective than simple instance weighting for this data.

## 4. Skills Demonstrated

* **Industrial Signal Processing:** Time-series feature generation for high-frequency sensor data.
* **Supervised Learning:** Regression, Tree Ensembles (Random Forest, XGBoost), and Deep Learning (LSTM).
* **Model Interpretation:** Permutation importance, clustering, and regime discovery.
* **Domain Adaptation:** Analysis of model performance under simulated domain shifts.
* **Experimental Design:** End-to-end evaluation and documentation for production-line quality prediction.

---

*This project was developed for academic and research purposes to demonstrate advanced predictive maintenance techniques.*
