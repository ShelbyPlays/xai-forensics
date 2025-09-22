import pandas as pd, joblib, shap
from pathlib import Path

FEA = Path("data/processed/features.parquet")
MODEL = Path("models/xgb_model.joblib")

def to_sentence(name, value):
    mapping = {
        "fails": f"repeated failed logins ({value})",
        "offhour_events": f"activity during off-hours ({value})",
        "uniq_ips": f"unusual number of source IPs ({value})",
        "logins": f"many login attempts ({value})",
        "events": f"high event volume ({value})",
        "successes": f"successful logins ({value})"
    }
    return mapping.get(name, f"{name}={value}")

def main():
    df = pd.read_parquet(FEA)
    X = df[["events","logins","fails","successes","offhour_events","uniq_ips"]]
    row = X.sample(1, random_state=11)
    clf = joblib.load(MODEL)
    proba = clf.predict_proba(row)[:,1][0]

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(row)
    contrib = sorted(zip(row.columns, sv[0]), key=lambda t: abs(t[1]), reverse=True)[:3]

    reasons = [to_sentence(n, int(row.iloc[0][n])) for n,_ in contrib]
    print("\n=== Case Report ===")
    print(f"Model decision: Anomalous probability = {proba:.2f}")
    print("Top reasons:", "; ".join(reasons))
    print("Error-rate note: See reports/metrics.json for FPR/FNR on held-out set.\n")

if __name__ == "__main__":
    main()
