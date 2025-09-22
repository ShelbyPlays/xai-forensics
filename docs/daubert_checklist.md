# Daubert Checklist – XAI Forensics

**Testability**  
- Code and data pipeline are reproducible via scripts (un_pipeline.ps1).
- Model artifact: models/xgb_model.joblib.

**Known Error Rate**  
- See \eports/metrics.json\ (precision/recall, confusion matrix).
- Document test split and seed.

**Peer Review & Acceptance**  
- Explanations via SHAP (well-cited, peer-reviewed).
- Model: XGBoost (widely used).

**Standards & Controls**  
- Feature schema is fixed: events, logins, fails, successes, offhour_events, uniq_ips.
- Version locking in \equirements.lock.txt\.

**Reproducibility**  
- \un_pipeline.ps1\ regenerates data → features → model → case report.
- \src/explain_shap.py\ and \src/explain_local.py\ regenerate plots/explanations.

**Chain of Custody (if using real logs)**  
- Keep raw logs immutable, hash artifacts, and record timestamps.