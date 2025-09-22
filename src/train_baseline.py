import pandas as pd, json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import numpy as np

FEA = Path("data/processed/features.parquet")
MODEL = Path("models/xgb_model.joblib")
METRICS = Path("reports/metrics.json")

FEATURES = ["events","logins","fails","successes","offhour_events","uniq_ips"]

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have both classes. If not, derive labels from a simple risk score."""
    if "label" not in df.columns:
        df["label"] = 0

    vc = df["label"].value_counts()
    if len(vc) < 2:
        # Build a simple risk score and label top 20% as positive
        score = 0.6 * df["fails"] + 0.4 * df["offhour_events"]
        thresh = np.quantile(score, 0.80)
        df["label"] = (score >= thresh).astype(int)
        vc = df["label"].value_counts()
        print(f"[Info] Original labels had one class; re-labeled using risk score. New class counts: {vc.to_dict()}")
    else:
        print(f"[Info] Class counts: {vc.to_dict()}")
    return df

def main():
    df = pd.read_parquet(FEA)

    # Make sure we have usable labels
    df = ensure_labels(df)

    X = df[FEATURES]
    y = df["label"]

    # If minority class is tiny, shrink test size + avoid stratify edge-cases
    test_size = 0.2 if y.value_counts().min() >= 2 else 0.1
    stratify = y if len(y.value_counts()) == 2 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=7, test_size=test_size, stratify=stratify
    )

    # Guard again in case the split still produced a single-class set
    if y_train.nunique() < 2:
        # Move one positive to train if possible
        pos_idx = y[y == 1].index.tolist()
        neg_idx = y[y == 0].index.tolist()
        if pos_idx and neg_idx:
            # Force at least one of each into train
            X_train = pd.concat([X_train, X.loc[[pos_idx[0]]]])
            y_train = pd.concat([y_train, y.loc[[pos_idx[0]]]])
            # If train had only positives, add a negative; vice-versa
            if (y_train == 1).all() and neg_idx:
                X_train = pd.concat([X_train, X.loc[[neg_idx[0]]]])
                y_train = pd.concat([y_train, y.loc[[neg_idx[0]]]])
            if (y_train == 0).all() and pos_idx:
                X_train = pd.concat([X_train, X.loc[[pos_idx[0]]]])
                y_train = pd.concat([y_train, y.loc[[pos_idx[0]]]])
        print("[Info] Adjusted training set to include both classes.")

    clf = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9,
        random_state=7, n_jobs=4, eval_metric="logloss"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS,"w") as f:
        json.dump({"report":rep, "confusion_matrix":cm}, f, indent=2)

    print("Saved model & metrics.")
    print("Confusion matrix:", cm)

if __name__ == "__main__":
    main()
