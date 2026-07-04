"""
collector.py
------------------------------------------------------------
Lightweight version of your data-fetch + risk-scoring logic,
designed to run unattended on GitHub Actions every few hours.

No email, no PDF, no matplotlib — just: fetch real NOAA data,
score risk (same formula as solarshield_v6.py, LEO orbit),
append one row to live_data.csv.

This file lives in your repo at the root. The GitHub Actions
workflow (.github/workflows/collect_data.yml) runs this on a
schedule and commits the updated CSV automatically.
------------------------------------------------------------
"""

import urllib.request
import json
import csv
import os
from datetime import datetime, timezone

OUT_FILE = "live_data.csv"


def fetch(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SolarShieldAI/6.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception:
        return None


def get_solar_data():
    bz = bt = speed = density = 0.0
    current_kp = max_kp = ssn = f107 = 0.0
    alert_count = 0

    mag = fetch("https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json")
    if not mag or len(mag) <= 1:
        mag = fetch("https://services.swpc.noaa.gov/products/solar-wind/mag-6-hour.json")
    if mag and len(mag) > 1:
        try: bz = float(mag[-1][3])
        except Exception: pass
        try: bt = float(mag[-1][6])
        except Exception: pass

    plasma = fetch("https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json")
    if not plasma or len(plasma) <= 1:
        plasma = fetch("https://services.swpc.noaa.gov/products/solar-wind/plasma-6-hour.json")
    if plasma and len(plasma) > 1:
        try: speed = float(plasma[-1][2])
        except Exception: pass
        try: density = float(plasma[-1][1])
        except Exception: pass

    kp1m = fetch("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
    if kp1m and len(kp1m) > 0:
        try:
            current_kp = float(kp1m[-1].get("kp_index", kp1m[-1].get("kp", 0)))
        except Exception:
            pass
    if current_kp == 0:
        kpn = fetch("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
        if kpn and len(kpn) > 1:
            try:
                last_row = kpn[-1]
                current_kp = float(last_row.get("Kp", last_row.get("kp", 0)))
            except Exception:
                pass

    kp_fc = fetch("https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json")
    if kp_fc and len(kp_fc) > 1:
        try:
            vals = []
            for r in kp_fc[1:9]:
                v = r.get("kp", r.get("Kp")) if isinstance(r, dict) else None
                if v is not None: vals.append(float(v))
            max_kp = max(vals) if vals else 0
        except Exception:
            pass

    raw = fetch("https://services.swpc.noaa.gov/products/alerts.json")
    if raw:
        alert_count = min(len(raw), 20)

    cycle = fetch("https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json")
    if cycle and len(cycle) > 0:
        try:
            ssn = float(cycle[-1]["ssn"])
            f107 = float(cycle[-1]["f10.7"])
        except Exception:
            pass

    return {
        "bz": bz, "bt": bt, "speed": speed, "density": density,
        "current_kp": current_kp, "max_kp": max_kp,
        "alert_count": alert_count, "ssn": ssn, "f107": f107
    }


def calc_risk_leo(d):
    """Same scoring logic as solarshield_v6.py's calc_risk(), LEO branch only."""
    bz, speed, kp, max_kp, alerts, ssn = (
        d["bz"], d["speed"], d["current_kp"], d["max_kp"], d["alert_count"], d["ssn"]
    )
    score = 0
    if bz < -20: score += 4
    elif bz < -10: score += 3
    elif bz < -5: score += 2
    elif bz < 0: score += 1

    if speed > 700: score += 3
    elif speed > 500: score += 2
    elif speed > 400: score += 1

    if kp >= 7: score += 4
    elif kp >= 5: score += 3
    elif kp >= 4: score += 2
    elif kp >= 3: score += 1

    if max_kp >= 6: score += 2
    elif max_kp >= 5: score += 1

    if alerts >= 5: score += 2
    elif alerts >= 2: score += 1

    if ssn > 150: score += 2
    elif ssn > 100: score += 1

    drag_f = (1 + (kp * 0.15)) * (1 + max(0, (speed - 350) / 1000))
    if (drag_f - 1) * 100 > 100: score += 1

    score = min(10, score)
    if score >= 8: level = "CRITICAL"
    elif score >= 6: level = "HIGH"
    elif score >= 4: level = "MODERATE"
    elif score >= 2: level = "LOW"
    else: level = "VERY LOW"

    return score, level


def main():
    d = get_solar_data()
    score, level = calc_risk_leo(d)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    file_exists = os.path.exists(OUT_FILE)
    with open(OUT_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "kp", "bz", "bt", "solar_wind_speed",
                        "density", "max_kp_forecast", "ssn", "f107",
                        "active_alerts", "risk_score", "risk_level"])
        w.writerow([now_str, d["current_kp"], d["bz"], d["bt"], d["speed"],
                    d["density"], d["max_kp"], d["ssn"], d["f107"],
                    d["alert_count"], score, level])

    print(f"{now_str} | Kp={d['current_kp']} | risk={score}/10 ({level})")


if __name__ == "__main__":
    main()
