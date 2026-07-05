"""
collector.py  (DEBUG VERSION)
------------------------------------------------------------
Same as your original, but every fetch() call now prints:
  - the URL it tried
  - HTTP status / response length on success
  - the FULL exception type + message on failure

This is temporary. Once we see the real error in the GitHub
Actions log, we fix the root cause and can strip the logging
back out (or leave it — it's harmless either way).
------------------------------------------------------------
"""

import urllib.request
import urllib.error
import json
import csv
import os
import sys
import time
from datetime import datetime, timezone

OUT_FILE = "live_data.csv"


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.swpc.noaa.gov/",
}


def fetch(url, retries=2, delay=3):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=BROWSER_HEADERS)
            with urllib.request.urlopen(req, timeout=15) as r:
                status = r.status
                raw = r.read()
                data = json.loads(raw)
                print(f"[OK]   {url}  status={status}  rows={len(data) if isinstance(data, list) else 'n/a'}  (attempt {attempt})", flush=True)
                return data
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode(errors="replace")[:300]
            except Exception:
                pass
            print(f"[FAIL] {url}  HTTPError  code={e.code}  reason={e.reason}  attempt={attempt}  body={body!r}", flush=True)
            last_err = e
        except urllib.error.URLError as e:
            print(f"[FAIL] {url}  URLError  reason={e.reason}  attempt={attempt}", flush=True)
            last_err = e
        except Exception as e:
            print(f"[FAIL] {url}  {type(e).__name__}: {e}  attempt={attempt}", flush=True)
            last_err = e
        if attempt < retries:
            time.sleep(delay)
    return None



def get_solar_data():
    bz = bt = speed = density = 0.0
    current_kp = max_kp = ssn = f107 = 0.0
    alert_count = 0

    mag = fetch("https://services.swpc.noaa.gov/products/solar-wind/mag-2-hour.json")
    if not mag or len(mag) <= 1:
        print("  -> mag-2-hour empty/failed, trying mag-6-hour fallback", flush=True)
        mag = fetch("https://services.swpc.noaa.gov/products/solar-wind/mag-6-hour.json")
    if not mag or len(mag) <= 1:
        print("  -> mag-6-hour also failed, trying mag-1-day fallback", flush=True)
        mag = fetch("https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json")
    if mag and len(mag) > 1:
        try:
            bz = float(mag[-1][3])
        except Exception as e:
            print(f"  -> bz parse failed on row {mag[-1]}: {e}", flush=True)
        try:
            bt = float(mag[-1][6])
        except Exception as e:
            print(f"  -> bt parse failed on row {mag[-1]}: {e}", flush=True)
    else:
        print("  -> BOTH mag endpoints failed, bz/bt staying at 0.0", flush=True)

    plasma = fetch("https://services.swpc.noaa.gov/products/solar-wind/plasma-2-hour.json")
    if not plasma or len(plasma) <= 1:
        print("  -> plasma-2-hour empty/failed, trying plasma-6-hour fallback", flush=True)
        plasma = fetch("https://services.swpc.noaa.gov/products/solar-wind/plasma-6-hour.json")
    if not plasma or len(plasma) <= 1:
        print("  -> plasma-6-hour also failed, trying plasma-1-day fallback", flush=True)
        plasma = fetch("https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json")
    if plasma and len(plasma) > 1:
        try:
            speed = float(plasma[-1][2])
        except Exception as e:
            print(f"  -> speed parse failed on row {plasma[-1]}: {e}", flush=True)
        try:
            density = float(plasma[-1][1])
        except Exception as e:
            print(f"  -> density parse failed on row {plasma[-1]}: {e}", flush=True)
    else:
        print("  -> BOTH plasma endpoints failed, speed/density staying at 0.0", flush=True)

    kp1m = fetch("https://services.swpc.noaa.gov/json/planetary_k_index_1m.json")
    if kp1m and len(kp1m) > 0:
        try:
            current_kp = float(kp1m[-1].get("kp_index", kp1m[-1].get("kp", 0)))
        except Exception as e:
            print(f"  -> kp1m parse failed: {e}", flush=True)
    if current_kp == 0:
        kpn = fetch("https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json")
        if kpn and len(kpn) > 1:
            try:
                last_row = kpn[-1]
                current_kp = float(last_row.get("Kp", last_row.get("kp", 0)))
            except Exception as e:
                print(f"  -> kpn parse failed: {e}", flush=True)

    kp_fc = fetch("https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json")
    if kp_fc and len(kp_fc) > 1:
        try:
            vals = []
            for r in kp_fc[1:9]:
                v = r.get("kp", r.get("Kp")) if isinstance(r, dict) else None
                if v is not None:
                    vals.append(float(v))
            max_kp = max(vals) if vals else 0
        except Exception as e:
            print(f"  -> kp_fc parse failed: {e}", flush=True)

    raw = fetch("https://services.swpc.noaa.gov/products/alerts.json")
    if raw:
        alert_count = min(len(raw), 20)

    cycle = fetch("https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json")
    if cycle and len(cycle) > 0:
        try:
            ssn = float(cycle[-1]["ssn"])
            f107 = float(cycle[-1]["f10.7"])
        except Exception as e:
            print(f"  -> cycle parse failed: {e}", flush=True)

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
    print("=" * 60, flush=True)
    print(f"SolarShield collector run @ {datetime.now(timezone.utc).isoformat()}", flush=True)
    print("=" * 60, flush=True)

    d = get_solar_data()
    score, level = calc_risk_leo(d)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    print("-" * 60, flush=True)
    print(f"FINAL VALUES: bz={d['bz']} bt={d['bt']} speed={d['speed']} density={d['density']} kp={d['current_kp']}", flush=True)
    print("-" * 60, flush=True)

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

    print(f"{now_str} | Kp={d['current_kp']} | risk={score}/10 ({level})", flush=True)


if __name__ == "__main__":
    main()
