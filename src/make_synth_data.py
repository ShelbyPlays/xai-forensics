import pandas as pd, numpy as np
from pathlib import Path

RAW = Path("data/raw/logs.csv")

def main():
    rng = np.random.default_rng(42)
    users = ["alice","bob","carol","dave"]
    rows = []
    for day in pd.date_range("2025-01-01","2025-01-07",freq="D"):
        for u in users:
            n = rng.integers(200,300)
            times = pd.date_range(day, day+pd.Timedelta(days=1)-pd.Timedelta(seconds=1), periods=n)
            ips = [f"10.0.{rng.integers(0,4)}.{rng.integers(1,255)}" for _ in range(n)]
            events = rng.choice(["login","logout","passwd_change","ssh_open"], size=n, p=[0.6,0.25,0.05,0.10])
            success = (rng.random(n) > 0.15).astype(int)  # ~15% fails
            # inject anomalies
            if rng.random() < 0.4:
                for _ in range(rng.integers(10,25)):
                    rows.append([times[rng.integers(0,n)], u, f"203.0.113.{rng.integers(1,255)}", "login", 0])
            rows += list(map(list, zip(times, [u]*n, ips, events, success)))
    df = pd.DataFrame(rows, columns=["timestamp","user","src_ip","event","success"]).sort_values("timestamp")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW, index=False)
    print("Wrote", RAW, len(df))

if __name__ == "__main__":
    main()
