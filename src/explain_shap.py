import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
FEA = Path("data/processed/features.parquet")
MODEL = Path("models/xgb_model.joblib")
OUT = Path("reports/figures")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    # Load data and model
    df = pd.read_parquet(FEA)
    X = df[["events", "logins", "fails", "successes", "offhour_events", "uniq_ips"]]
    model = joblib.load(MODEL)

    # SHAP explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global summary (bar chart)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary_bar.png", dpi=200)
    plt.close()

    # Global beeswarm plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_beeswarm.png", dpi=200)
    plt.close()

    print("✅ SHAP plots saved to:", OUT)

if __name__ == "__main__":
    main()
