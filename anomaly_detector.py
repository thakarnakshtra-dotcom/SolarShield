"""
anomaly_detector.py
------------------------------------------------------------
Statistical anomaly detection, independent of the Kp>=5 storm
threshold used elsewhere. Catches "unusual for the recent
baseline" conditions even when nothing has officially crossed
into storm territory -- the kind of subtle drift a tired human
scanning one number would miss.

Method: rolling z-score against the last N real readings in
live_data.csv (default N=60, ~5 days at 2-hour intervals).
Deliberately simple and explainable (no black-box model) --
"why did it flag this" needs a one-line honest answer.
------------------------------------------------------------
"""

import csv
import os
import statistics

OUT_FILE = "live_data.csv"
Z_SCORE_THRESHOLD = 2.5
MIN_HISTORY_ROWS = 15
HISTORY_WINDOW = 60


def _load_recent_values(field, n=HISTORY_WINDOW, csv_path=OUT_FILE):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    vals = []
    for r in rows[-n:]:
        try:
            vals.append(float(r.get(field, "")))
        except (ValueError, TypeError):
            continue
    return vals


def check_all(current_reading, csv_path=OUT_FILE):
    """
    current_reading: dict, e.g. {"kp": 4.0, "bz": -8.2,
                                  "solar_wind_speed": 610, "density": 3.2}
    Returns: dict keyed by field name ->
        {"z_score": float, "is_anomaly": bool,
         "baseline_mean": float, "baseline_std": float}
    Fields with insufficient history are omitted.
    """
    results = {}
    for field, value in current_reading.items():
        if value is None:
            continue
        history = _load_recent_values(field, csv_path=csv_path)
        if len(history) < MIN_HISTORY_ROWS:
            continue
        mean = statistics.mean(history)
        std = statistics.pstdev(history)
        if std == 0:
            continue
        z = (value - mean) / std
        results[field] = {
            "z_score": round(z, 2),
            "is_anomaly": abs(z) >= Z_SCORE_THRESHOLD,
            "baseline_mean": round(mean, 3),
            "baseline_std": round(std, 3),
        }
    return results


if __name__ == "__main__":
    test_reading = {"kp": 6.0, "bz": -15.0, "solar_wind_speed": 750, "density": 1.0}
    result = check_all(test_reading)
    if not result:
        print("Not enough history yet to compute a baseline (need >=15 real rows in live_data.csv).")
    else:
        for field, r in result.items():
            flag = "ANOMALY" if r["is_anomaly"] else "normal"
            print(f"  {field}: z={r['z_score']}  ({flag}, baseline_mean={r['baseline_mean']})")
