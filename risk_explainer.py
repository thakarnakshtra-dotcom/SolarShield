"""
risk_explainer.py
------------------------------------------------------------
SHAP-style explainability layer for calc_risk_leo() in collector.py.

Your rule-based risk score is already deterministic and additive
(each factor adds a fixed number of points) -- which means we can
get PERFECT, exact attribution for free. This isn't an approximation
like real SHAP values on a black-box model; it's the literal true
breakdown, because the underlying model is transparent by design.
That's actually a selling point: "our rule engine is explainable
by construction, no post-hoc approximation needed."

Usage as a library:
    from risk_explainer import explain_risk
    result = explain_risk(d)   # d = same dict calc_risk_leo() takes
    print(result["summary_text"])

Usage standalone (reads the latest row of live_data.csv):
    python3 risk_explainer.py
------------------------------------------------------------
"""

import csv
import json
import os

LIVE_FILE = "live_data.csv"


def explain_risk(d):
    """
    d: dict with bz, speed, current_kp, max_kp, alert_count, ssn
       (identical shape to what calc_risk_leo() expects)

    Returns a dict:
        total_score, risk_level, factors (list, sorted by points desc),
        ml_context (note about what the ML model does/doesn't see),
        summary_text (ready-to-display string)
    """
    bz, speed, kp, max_kp, alerts, ssn = (
        d["bz"], d["speed"], d["current_kp"], d["max_kp"],
        d["alert_count"], d["ssn"],
    )

    factors = []

    # --- Bz (southward IMF component) ---
    if bz < -20:
        pts, why = 4, f"Bz={bz:.1f} nT is strongly southward (< -20)"
    elif bz < -10:
        pts, why = 3, f"Bz={bz:.1f} nT is strongly southward (< -10)"
    elif bz < -5:
        pts, why = 2, f"Bz={bz:.1f} nT is moderately southward (< -5)"
    elif bz < 0:
        pts, why = 1, f"Bz={bz:.1f} nT is weakly southward (< 0)"
    else:
        pts, why = 0, f"Bz={bz:.1f} nT is northward/neutral, no contribution"
    factors.append({"factor": "Bz (IMF orientation)", "points": pts, "why": why})

    # --- Solar wind speed ---
    if speed > 700:
        pts, why = 3, f"Solar wind speed={speed:.0f} km/s is very high (> 700)"
    elif speed > 500:
        pts, why = 2, f"Solar wind speed={speed:.0f} km/s is elevated (> 500)"
    elif speed > 400:
        pts, why = 1, f"Solar wind speed={speed:.0f} km/s is above baseline (> 400)"
    else:
        pts, why = 0, f"Solar wind speed={speed:.0f} km/s is near baseline"
    factors.append({"factor": "Solar wind speed", "points": pts, "why": why})

    # --- Current Kp ---
    if kp >= 7:
        pts, why = 4, f"Current Kp={kp:.1f} is severe storm level (>= 7)"
    elif kp >= 5:
        pts, why = 3, f"Current Kp={kp:.1f} is storm level (>= 5)"
    elif kp >= 4:
        pts, why = 2, f"Current Kp={kp:.1f} is active/unsettled (>= 4)"
    elif kp >= 3:
        pts, why = 1, f"Current Kp={kp:.1f} is slightly unsettled (>= 3)"
    else:
        pts, why = 0, f"Current Kp={kp:.1f} is quiet"
    factors.append({"factor": "Current Kp index", "points": pts, "why": why})

    # --- Forecast max Kp ---
    if max_kp >= 6:
        pts, why = 2, f"Forecast max Kp={max_kp:.1f} over next ~24h is HIGH (>= 6)"
    elif max_kp >= 5:
        pts, why = 1, f"Forecast max Kp={max_kp:.1f} over next ~24h is elevated (>= 5)"
    else:
        pts, why = 0, f"Forecast max Kp={max_kp:.1f} is not concerning"
    factors.append({"factor": "Forecast max Kp", "points": pts, "why": why})

    # --- Active NOAA alerts ---
    if alerts >= 5:
        pts, why = 2, f"{alerts} active NOAA space weather alerts (>= 5)"
    elif alerts >= 2:
        pts, why = 1, f"{alerts} active NOAA space weather alerts (>= 2)"
    else:
        pts, why = 0, f"{alerts} active NOAA alerts, not significant"
    factors.append({"factor": "Active NOAA alerts", "points": pts, "why": why})

    # --- Sunspot number (solar cycle activity) ---
    if ssn > 150:
        pts, why = 2, f"Sunspot number={ssn:.0f} indicates high solar activity (> 150)"
    elif ssn > 100:
        pts, why = 1, f"Sunspot number={ssn:.0f} indicates elevated solar activity (> 100)"
    else:
        pts, why = 0, f"Sunspot number={ssn:.0f} is not a major contributor"
    factors.append({"factor": "Sunspot number (SSN)", "points": pts, "why": why})

    # --- Atmospheric drag factor (LEO-specific, derived) ---
    drag_f = (1 + (kp * 0.15)) * (1 + max(0, (speed - 350) / 1000))
    drag_pct = (drag_f - 1) * 100
    if drag_pct > 100:
        pts, why = 1, f"Estimated LEO drag increase is {drag_pct:.0f}% above baseline (> 100%)"
    else:
        pts, why = 0, f"Estimated LEO drag increase is {drag_pct:.0f}% above baseline (not yet significant)"
    factors.append({"factor": "Atmospheric drag (LEO)", "points": pts, "why": why})

    raw_total = sum(f["points"] for f in factors)
    total_score = min(10, raw_total)

    if total_score >= 8:
        level = "CRITICAL"
    elif total_score >= 6:
        level = "HIGH"
    elif total_score >= 4:
        level = "MODERATE"
    elif total_score >= 2:
        level = "LOW"
    else:
        level = "VERY LOW"

    # Sort by contribution, biggest first; drop zero-point factors from the
    # headline list but keep them available for a full breakdown if needed.
    factors_sorted = sorted(factors, key=lambda f: f["points"], reverse=True)
    contributing = [f for f in factors_sorted if f["points"] > 0]

    ml_context = (
        "Note: the separate ML storm-probability model (predict_storm.py) "
        "only uses Kp history (current + 2 prior readings, slope, rolling "
        "mean) -- it does not see Bz, solar wind speed, alerts, or SSN. So "
        "it can legitimately disagree with this rule-based score, especially "
        "when Bz/speed/alerts are driving the risk rather than Kp itself."
    )

    if contributing:
        lines = [f"  - {f['factor']}: +{f['points']} pts — {f['why']}" for f in contributing]
        breakdown_text = "\n".join(lines)
    else:
        breakdown_text = "  - No individual factor is currently elevated."

    summary_text = (
        f"Risk score: {total_score}/10 ({level})\n"
        f"Contributing factors (largest first):\n{breakdown_text}\n\n"
        f"{ml_context}"
    )

    return {
        "total_score": total_score,
        "risk_level": level,
        "factors": factors_sorted,
        "ml_context": ml_context,
        "summary_text": summary_text,
    }


def _latest_row_as_dict(path=LIVE_FILE):
    """Reads the last row of live_data.csv and reshapes it into the dict
    shape explain_risk()/calc_risk_leo() expect."""
    with open(path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    r = rows[-1]
    return {
        "bz": float(r.get("bz", 0) or 0),
        "speed": float(r.get("solar_wind_speed", 0) or 0),
        "current_kp": float(r.get("kp", 0) or 0),
        "max_kp": float(r.get("max_kp_forecast", 0) or 0),
        "alert_count": int(float(r.get("active_alerts", 0) or 0)),
        "ssn": float(r.get("ssn", 0) or 0),
    }, r.get("timestamp", "unknown")


if __name__ == "__main__":
    if not os.path.exists(LIVE_FILE):
        print(f"{LIVE_FILE} not found in this directory.")
    else:
        d, ts = _latest_row_as_dict()
        result = explain_risk(d)
        print(f"Latest reading: {ts}\n")
        print(result["summary_text"])
        print("\nFull JSON (for wiring into an API/chatbot):")
        print(json.dumps(result, indent=2))
