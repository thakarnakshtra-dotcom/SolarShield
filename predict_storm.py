"""
predict_storm.py
------------------------------------------------------------
Loads the trained logistic regression weights from
model_weights.json (produced by train_and_export.py) and
scores a live sequence of Kp readings for storm probability.

This is the "real ML" layer sitting on top of your existing
rule-based calc_risk_leo() in collector.py -- it doesn't
replace the rule-based score, it adds a second, independently
validated signal alongside it. Showing both side-by-side in
your pitch is more credible than replacing a working system
outright.

Usage as a library:
    from predict_storm import StormPredictor
    sp = StormPredictor("model_weights.json")
    prob = sp.predict(kp_now=4.0, kp_1_ago=3.0, kp_2_ago=2.0)

Usage standalone (for a quick sanity check):
    python3 predict_storm.py 4.0 3.0 2.0
------------------------------------------------------------
"""

import json
import math
import os
import sys


class StormPredictor:
    def __init__(self, weights_path="model_weights.json"):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"{weights_path} not found. Run train_and_export.py first "
                "to generate it from real historical data."
            )
        with open(weights_path, "r") as f:
            self.model = json.load(f)
        self.weights = self.model["weights"]
        self.thresholds = self.model["recommended_thresholds"]

    def predict(self, kp_now, kp_1_ago, kp_2_ago):
        """
        Returns probability (0-1) that Kp will reach storm level
        (>= 5.0) within the model's lookahead window (default: next
        2 x 3-hour readings, i.e. ~6 hours ahead).
        """
        slope = kp_now - kp_2_ago
        rolling_mean = (kp_now + kp_1_ago + kp_2_ago) / 3.0
        x = [1.0, kp_now, kp_1_ago, slope, rolling_mean]
        z = sum(w * xi for w, xi in zip(self.weights, x))
        z = max(-30, min(30, z))  # avoid overflow
        return 1.0 / (1.0 + math.exp(-z))

    def classify(self, kp_now, kp_1_ago, kp_2_ago, sensitivity="balanced"):
        """
        sensitivity: 'high_precision' | 'balanced' | 'high_recall'
        Returns (probability, is_alert: bool, threshold_used: float)
        """
        threshold = self.thresholds.get(sensitivity, 0.25)
        prob = self.predict(kp_now, kp_1_ago, kp_2_ago)
        return prob, prob >= threshold, threshold

    def model_info(self):
        return {
            "trained_at": self.model.get("trained_at"),
            "training_rows": self.model.get("training_rows"),
            "date_range": self.model.get("date_range"),
            "validated_test_metrics": self.model.get("validated_test_metrics_at_threshold_0.25"),
        }


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 predict_storm.py <kp_now> <kp_1_ago> <kp_2_ago>")
        sys.exit(1)

    kp_now, kp_1, kp_2 = map(float, sys.argv[1:4])
    sp = StormPredictor("model_weights.json")
    prob, alert, thresh = sp.classify(kp_now, kp_1, kp_2, sensitivity="balanced")

    print(f"Model info: {json.dumps(sp.model_info(), indent=2)}")
    print(f"\nInput: kp_now={kp_now}, kp_1_ago={kp_1}, kp_2_ago={kp_2}")
    print(f"Storm probability (next ~6h): {prob:.1%}")
    print(f"Alert (balanced threshold={thresh}): {'YES' if alert else 'no'}")
