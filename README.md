# Dynamic Predictive Modelling of a Hydraulic Production Line (Bosch Project)

This repository contains work for the Bosch R&D project **“Generalisation of Dynamic Predictive Modelling Across Production Line.”**  
It builds predictive models for component health and system stability from multi-sensor hydraulic time-series and studies how well these models generalise across product categories.

---

## 1. Dataset

- **Source:** UCI Machine Learning Repository –  
  **“Condition Monitoring of Hydraulic Systems”**  
  https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- Cycle-based time-series from multiple sensors:
  - Pressures (PS1–PS6), flows (FS1–FS2), temperatures (TS1–TS4), vibration (VS1),
    and efficiency / power channels (CE, CP, SE, EPS1).
- Five labels per cycle (from `profile.txt`):
  - **Cooler_Condition**, **Valve_Condition**, **Internal_Pump_Leakage**,  
    **Hydraulic_Accumulator**, **Stable_Flag (0/1)**.

---

## 2. Methodology (high-level)

1. **Data preparation**
   - Load all sensor matrices, align cycles, and join with the five labels into a single table (`df_final`).

2. **Feature engineering**
   - For every sensor & cycle, compute compact **time-domain** and **frequency-domain** features  
     (mean/std, RMS, crest factor, quantiles, lag-1 autocorr, entropy, skew/kurtosis, spectral energy, centroid, bandwidth, band powers).
   - Concatenate across sensors → wide tabular feature matrix.

3. **Domain shift simulation**
   - Create a synthetic `Product_Category` (Cat1 / Cat2) and apply controlled shifts to a subset of features in Cat2
     to mimic a new product / setting.

4. **Modelling**
   - **Baselines:** linear regression, polynomial regression, and one-feature curve fits.
   - **Feature-based ML:** RandomForestRegressor and XGBRegressor on engineered features.
   - **Sequence model:** LSTM on raw resampled sequences (n_cycles × timesteps × n_channels).

5. **Unsupervised analysis**
   - KMeans and DBSCAN on feature space; simple hidden-state labelling (e.g. “early_failure” regime).
   - Correlation and permutation-importance analyses for sensor/feature relevance.

---

## 3. Key Results & Insights

### 3.1 Predictive performance

- **Cooler_Condition**
  - Extremely predictable from engineered features: one-feature curve fit reaches **R² ≈ 0.998** (RMSE ≈ 1.77);  
    after snapping to nearest valid grade the error is effectively **0**.
- **Hydraulic_Accumulator**
  - Non-linear but learnable: one-feature curve fit reaches **R² ≈ 0.56**;  
    nearest-grade linear models also give high discrete performance.
- **Internal_Pump_Leakage & Valve_Condition**
  - Moderate signal: single-feature curve fits achieve **R² ≈ 0.76** and **≈ 0.50** respectively,  
    indicating strong but non-linear relationships with a few key features.
- **Stable_Flag**
  - Hardest label in simple baselines: best curve-fit baseline only reaches **R² ≈ 0.26**.
  - **Tree-based models on features perform much better:**  
    - XGBoost: **RMSE ≈ 0.128, MAE ≈ 0.040**  
    - Random Forest: **RMSE ≈ 0.146, MAE ≈ 0.052**
  - **LSTM on raw sequences underperforms** vs these feature-based models for Stable_Flag  
    (RMSE ≈ 0.243, MAE ≈ 0.135), suggesting that the engineered features already capture
    most of the useful temporal structure for this task.

### 3.2 Feature importance & regimes

- Permutation importance for **Stable_Flag** shows that:
  - Higher-order statistics of **PS2** (skew, kurtosis, peak-to-peak, crest factor),
    **FS1** (kurtosis, skew) and **EPS1** band powers (low/mid/high),
    plus **PS1** autocorrelation and band-high power,
  - are the most influential features, pointing to subtle changes in **pressure and flow waveform shape**
    rather than just mean levels.
- **KMeans clustering** on features finds:
  - One large, compact “good” cluster (~1959 cycles) and a smaller, noisy cluster (~246 cycles) with much higher within-cluster variance.
- A simple **“early_failure” hidden state** derived from clustering shows:
  - Lower average Cooler_Condition (~41 vs 100), mildly degraded Valve_Condition (~91),
    higher Internal_Pump_Leakage (~0.67) and reduced accumulator pressure (~107),
    capturing cycles in an **incipient degradation regime** rather than full failure.

### 3.3 Domain adaptation (Cat1 → Cat2)

- With a synthetic domain split (Cat1 = source, Cat2 = target):
  - **Zero-shot transfer** (train on Cat1, test on Cat2) already achieves **AUC ≈ 0.955** for Stable_Flag.
  - **Fine-tuning** (either freezing feature extractor and training only the head, or fine-tuning all layers)
    produces very similar AUC/ACC/F1, suggesting the learned representations are fairly domain-robust.
  - A simple **re-weight + fine-tune** strategy performs poorly (AUC ≈ 0.51, F1 ≈ 0.01),
    highlighting that naïve re-weighting can harm performance more than it helps.

Overall, **feature-based tree models** provide strong, interpretable baselines, especially for stability prediction, while the analysis of feature importance, clustering and domain shift offers a practical roadmap for adapting Bosch’s existing quality-prediction tool to new lines and products.

---

## 4. Skills Demonstrated

- Time-series signal processing and feature generation for industrial sensor data  
- Supervised learning with regression, tree ensembles, gradient boosting and LSTM sequence models  
- Permutation importance, clustering and regime discovery for model interpretation  
- Domain-shift analysis and simple domain-adaptation strategies  
- End-to-end experiment design, evaluation, and documentation in a production-line R&D context
