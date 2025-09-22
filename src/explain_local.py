import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

FEA = Path("data/processed/features.parquet")
MODEL = Path("models/xgb_model.joblib")
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["events","logins","fails","successes","offhour_events","uniq_ips"]
PLAIN_MAP = {
    "events": "High overall event volume",
    "logins": "Many login attempts",
    "fails": "Repeated failed logins",
    "successes": "Many successful logins",
    "offhour_events": "Activity during off-hours",
    "uniq_ips": "Unusual number of source IPs"
}

def pick_index(X, model, idx_arg):
    if idx_arg is not None:
        i = int(idx_arg)
        if not (0 <= i < len(X)):
            raise ValueError(f"--idx {i} out of range 0..{len(X)-1}")
        return i, float(model.predict_proba(X.iloc[[i]])[0,1])
    # default: pick the most anomalous row by predicted probability
    proba = model.predict_proba(X)[:,1]
    i = int(np.argmax(proba))
    return i, float(proba[i])

def main():
    parser = argparse.ArgumentParser(description="Create local SHAP plots for one case.")
    parser.add_argument("--idx", type=int, default=None, help="Row index to explain (default: highest predicted anomaly).")
    args = parser.parse_args()

    if not FEA.exists():
        raise FileNotFoundError(f"Missing {FEA}. Run featurize first.")
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing {MODEL}. Train the model first.")

    df = pd.read_parquet(FEA)
    X = df[FEATURES]
    model = joblib.load(MODEL)

    # choose case
    idx, proba = pick_index(X, model, args.idx)
    x_row = X.iloc[[idx]]

    # SHAP values for a single row
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x_row)  # shape (1, n_features)
    base_value = float(explainer.expected_value)

    # ✅ Waterfall plot (new API)
    exp = shap.Explanation(
        values=shap_vals[0],
        base_values=base_value,
        data=x_row.values[0],
        feature_names=FEATURES
    )
    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=10)
    out_waterfall = OUT / f"local_waterfall_idx{idx}.png"
    plt.tight_layout()
    plt.savefig(out_waterfall, dpi=220, bbox_inches="tight")
    plt.close()

    # Decision plot (PNG)
    plt.figure()
    shap.decision_plot(
        base_value,
        shap_vals,
        x_row,
        feature_names=FEATURES,
        show=False
    )
    out_decision = OUT / f"local_decision_idx{idx}.png"
    plt.tight_layout()
    plt.savefig(out_decision, dpi=220, bbox_inches="tight")
    plt.close()

    # Top reasons JSON (by absolute contribution)
    contribs = [
        {"feature": f,
         "value": float(x_row.iloc[0][f]),
         "contribution": float(shap_vals[0,i]),
         "reason": PLAIN_MAP.get(f, f)}
        for i,f in enumerate(FEATURES)
    ]
    contribs_sorted = sorted(contribs, key=lambda d: abs(d["contribution"]), reverse=True)
    top3 = contribs_sorted[:3]

    summary = {
        "idx": idx,
        "probability": round(proba, 4),
        "base_value": round(base_value, 6),
        "top_reasons": top3,
        "all_contributions": contribs_sorted
    }
    out_json = OUT / f"local_explanation_idx{idx}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console summary
    print(f"Explained row idx={idx} | probability={proba:.2f}")
    print("Top reasons:")
    for r in top3:
        print(f" - {r['reason']} | value={r['value']} | SHAP={r['contribution']:+.4f}")
    print("Saved:")
    print(" ", out_waterfall)
    print(" ", out_decision)
    print(" ", out_json)

if __name__ == "__main__":
    main()
