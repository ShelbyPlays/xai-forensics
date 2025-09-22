# XAI Forensics – Overview

**Goal:** Explainable anomaly detection from logs using 6 transparent features.

**Endpoints:**
- GET /health
- POST /featurize  (raw logs → features)
- POST /predict    (features → label + probability)
- POST /explain    (features → SHAP reasons)

**Artifacts to cite:**
- reports/metrics.json (error rates)
- reports/figures/shap_summary_bar.png
- reports/figures/shap_beeswarm.png
- reports/figures/local_waterfall_idxX.png
- reports/figures/local_decision_idxX.png