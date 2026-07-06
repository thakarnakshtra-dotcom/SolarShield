import sys
from backtest_model import load_data, resample, build_features, train_logistic_regression, evaluate, RESAMPLE_HOURS, TRAIN_FRACTION

paths = sys.argv[1:] if len(sys.argv) > 1 else ["long_history_kp.csv"]
raw_rows = load_data(paths)
rows = resample(raw_rows, RESAMPLE_HOURS)
X, y, meta = build_features(rows)
split = int(len(X) * TRAIN_FRACTION)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
weights = train_logistic_regression(X_train, y_train)

print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>8} {'tp':>5} {'fp':>5} {'fn':>5}")
for t in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.05]:
    m = evaluate(weights, X_test, y_test, threshold=t)
    print(f"{t:>10.2f} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
