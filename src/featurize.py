import pandas as pd
from pathlib import Path

RAW = Path("data/raw/logs.csv")
OUT = Path("data/processed/features.parquet")

def off_hours(ts):
    # Works for both Series (use .dt.hour) and scalar Timestamp (.hour)
    if hasattr(ts, "dt"):
        h = ts.dt.hour
    else:
        h = ts.hour
    return (h < 6) | (h >= 22)

def main():
    df = pd.read_csv(RAW, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["off_hours"] = off_hours(df["timestamp"]).astype(int)
    df["failed_login"] = ((df["event"]=="login") & (df["success"]==0)).astype(int)
    df["success_login"] = ((df["event"]=="login") & (df["success"]==1)).astype(int)

    g = df.groupby(["user","date"])
    feats = g.agg(
        events=("event","count"),
        logins=("event", lambda s:(s=="login").sum()),
        fails=("failed_login","sum"),
        successes=("success_login","sum"),
        offhour_events=("off_hours","sum"),
        uniq_ips=("src_ip", pd.Series.nunique),
    ).reset_index()

    # label heuristic
    feats["label"] = ((feats["fails"] >= 8) & (feats["offhour_events"] >= 15)).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(OUT, index=False)
    print("Wrote", OUT, feats.shape)

if __name__ == "__main__":
    main()
