"""
train_and_export.py
------------------------------------------------------------
Trains the logistic regression model on ALL available real
historical data (long_history_kp.csv + historical_kp.csv) and
exports the learned weights + metadata to model_weights.json
so the live pipeline (predict_storm.py) can load them without
needing to retrain every time.

Also re-confirms honest held-out test metrics one more time
before exporting, so the exported model's documented
performance always matches what was actually validated.
------------------------------------------------------------
"""
import json
import sys
from datetime import datetime, timezone
from backtest_model import (
    load_data, resample, build_features, train_logistic_regression,
    evaluate, RESAMPLE_HOURS, TRAIN_FRACTION, STORM_THRESHOLD, LOOKAHEAD_STEPS
)

paths = ["long_history_kp.csv", "historical_kp.csv"]
raw_rows = load_data(paths)
rows = resample(raw_rows, RESAMPLE_HOURS)
X, y, meta = build_features(rows)

split = int(len(X) * TRAIN_FRACTION)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Final validation weights (trained on train split only, tested honestly)
val_weights = train_logistic_regression(X_train, y_train)
test_metrics = evaluate(val_weights, X_test, y_test, threshold=0.25)

# Production weights: retrain on ALL data (train+test) for deployment,
# since we've already validated the approach honestly above.
prod_weights = train_logistic_regression(X, y)

export = {
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "training_rows": len(X),
    "date_range": [str(rows[0][0]), str(rows[-1][0])],
    "storm_threshold_kp": STORM_THRESHOLD,
    "lookahead_steps": LOOKAHEAD_STEPS,
    "resample_hours": RESAMPLE_HOURS,
    "features": ["bias", "kp_now", "kp_1_ago", "slope_2step", "rolling_mean_3"],
    "weights": prod_weights,
    "validated_test_metrics_at_threshold_0.25": test_metrics,
    "recommended_thresholds": {
        "high_precision": 0.50,
        "balanced": 0.25,
        "high_recall": 0.15
    },
    "honesty_note": (
        "Validated on a real 70/30 time-based train/test split of "
        f"{len(X)} total 3-hour windows from GFZ Potsdam + NOAA data "
        "(July 2024 - July 2026). Production weights above are "
        "retrained on the FULL dataset for deployment; the "
        "'validated_test_metrics' field reflects the honest held-out "
        "performance from the train-only version, not the production "
        "weights directly (standard practice: validate on a split, "
        "then retrain on everything for final deployment)."
    )
}

with open("model_weights.json", "w") as f:
    json.dump(export, f, indent=2)

print("Exported model_weights.json")
print(json.dumps({k: v for k, v in export.items() if k != "weights"}, indent=2))
