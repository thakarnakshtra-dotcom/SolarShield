"""
backtest_model.py
------------------------------------------------------------
A genuine, honestly-evaluated storm prediction model, built on
top of REAL historical Kp data (from fetch_historical_kp.py).

WHAT IT PREDICTS:
  Given the current Kp reading and its recent trend, will Kp
  reach storm level (>= 5) within the next N readings
  (default N=2, i.e. within ~6 hours on the 3-hour feed)?

WHY THIS APPROACH (and not a fancier ML model):
  With only weeks of data, a complex model (LSTM etc.) will just
  memorize noise. A trend-based logistic model is the honest,
  defensible choice at this data volume — and it's a real
  step up from "linear trend on last 5 CSV rows", because it's
  backtested with a proper train/test split and reports actual
  precision/recall instead of an unvalidated guess.

  As your live collector accumulates more real data (weeks ->
  months), you can re-run this same script and watch accuracy
  actually improve, which is what "the model gets smarter over
  time" should really mean.

Usage:
  python3 backtest_model.py historical_kp.csv
------------------------------------------------------------
"""

import csv
import sys
import math
from datetime import datetime, timedelta

STORM_THRESHOLD = 5.0
LOOKAHEAD_STEPS = 2   # predict storm within next 2 readings (~6h after resampling)
TRAIN_FRACTION = 0.7
RESAMPLE_HOURS = 3    # matches Kp's native 3-hour resolution; fixes mixed-resolution data


def load_data(paths):
    """Accepts one or more CSV paths and merges them, deduping by timestamp."""
    if isinstance(paths, str):
        paths = [paths]
    merged = {}
    for path in paths:
        with open(path, "r") as f:
            for r in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
                    if ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)  # normalize to naive UTC
                    kp = float(r["kp"])
                    merged[ts] = kp
                except (KeyError, ValueError):
                    continue
    rows = sorted(merged.items())
    return rows


def resample(rows, hours=RESAMPLE_HOURS):
    """
    Your source data mixes resolutions (3-hour historical feed +
    1-minute live feed during an active storm). Feeding that straight
    into a fixed 'N steps ahead' model is invalid -- N steps means a
    different amount of real time depending on which chunk you're in,
    and a dense burst of near-duplicate minute readings can dominate
    the dataset. This buckets everything into uniform `hours`-sized
    windows (taking the max Kp seen in each window, since that's what
    matters for storm alerting) so every step means the same thing.
    """
    if not rows:
        return []
    bucketed = {}
    for ts, kp in rows:
        bucket_key = ts.replace(minute=0, second=0, microsecond=0)
        bucket_key = bucket_key.replace(hour=(bucket_key.hour // hours) * hours)
        bucketed.setdefault(bucket_key, []).append(kp)

    result = [(k, max(v)) for k, v in sorted(bucketed.items())]
    return result


def build_features(rows):
    """
    For each point t (with enough history and enough future),
    build features from the past and a binary label from the future.
    Features: current kp, kp 1-step ago, kp 2-steps ago,
              short-term slope, rolling mean of last 3.
    Label: 1 if any of the next LOOKAHEAD_STEPS readings >= STORM_THRESHOLD
    """
    X, y, meta = [], [], []
    kps = [r[1] for r in rows]
    n = len(kps)

    for i in range(2, n - LOOKAHEAD_STEPS):
        kp_now = kps[i]
        kp_1 = kps[i - 1]
        kp_2 = kps[i - 2]
        slope = kp_now - kp_2  # change over last 2 steps
        rolling_mean = (kp_now + kp_1 + kp_2) / 3.0

        future = kps[i + 1: i + 1 + LOOKAHEAD_STEPS]
        label = 1 if any(v >= STORM_THRESHOLD for v in future) else 0

        X.append([1.0, kp_now, kp_1, slope, rolling_mean])  # 1.0 = bias term
        y.append(label)
        meta.append(rows[i][0])

    return X, y, meta


def train_logistic_regression(X, y, lr=0.05, epochs=3000):
    """Plain gradient descent logistic regression, no external deps."""
    n_features = len(X[0])
    weights = [0.0] * n_features
    n = len(X)

    for _ in range(epochs):
        grad = [0.0] * n_features
        for xi, yi in zip(X, y):
            z = sum(w * x for w, x in zip(weights, xi))
            pred = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            error = pred - yi
            for j in range(n_features):
                grad[j] += error * xi[j]
        weights = [w - lr * (g / n) for w, g in zip(weights, grad)]

    return weights


def predict(weights, xi):
    z = sum(w * x for w, x in zip(weights, xi))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def evaluate(weights, X, y, threshold=0.5):
    tp = fp = tn = fn = 0
    for xi, yi in zip(X, y):
        p = predict(weights, xi)
        pred_label = 1 if p >= threshold else 0
        if pred_label == 1 and yi == 1:
            tp += 1
        elif pred_label == 1 and yi == 0:
            fp += 1
        elif pred_label == 0 and yi == 0:
            tn += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(y) if y else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1
    }


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["historical_kp.csv"]
    raw_rows = load_data(paths)
    rows = resample(raw_rows, RESAMPLE_HOURS)

    print(f"Loaded {len(raw_rows)} raw readings from: {', '.join(paths)}")
    print(f"Resampled to {len(rows)} uniform {RESAMPLE_HOURS}-hour buckets "
          f"(fixes the mixed-resolution issue)")
    if len(rows) < 30:
        print("WARNING: Very little data. Results below are not reliable yet.")
        print("Let the live collector run longer, then re-run this script.")

    X, y, meta = build_features(rows)
    if len(X) < 20:
        print(f"\nOnly {len(X)} usable training samples after feature engineering.")
        print("Need more historical data before backtest results mean anything.")
        return

    # time-based split (never shuffle time series data — that leaks the future)
    split = int(len(X) * TRAIN_FRACTION)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    storm_rate_train = sum(y_train) / len(y_train) if y_train else 0
    storm_rate_test = sum(y_test) / len(y_test) if y_test else 0

    print(f"\nTrain samples: {len(X_train)}  (storm-label rate: {storm_rate_train:.1%})")
    print(f"Test samples:  {len(X_test)}  (storm-label rate: {storm_rate_test:.1%})")

    if sum(y_train) == 0:
        print("\nNo storm events in training data at all — can't learn a storm")
        print("signal from quiet data alone. Need data that includes at least")
        print("one real storm period to validate this properly.")
        return

    weights = train_logistic_regression(X_train, y_train)

    print("\n--- TRAIN performance ---")
    train_metrics = evaluate(weights, X_train, y_train)
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n--- TEST performance (the honest number) ---")
    test_metrics = evaluate(weights, X_test, y_test)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nLearned weights [bias, kp_now, kp_1_ago, slope, rolling_mean]:")
    print("  " + ", ".join(f"{w:.4f}" for w in weights))

    print("\nNOTE: TEST metrics are the only honest measure of real-world")
    print("performance. TRAIN metrics will always look better and should")
    print("never be quoted as 'model accuracy' in a pitch or report.")


if __name__ == "__main__":
    main()
