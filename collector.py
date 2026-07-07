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

try:
    from predict_storm import StormPredictor
except ImportError:
    StormPredictor = None

try:
    from fetch_space_weather_plus import get_latest_flare, get_cme_earth_impact
except ImportError:
    get_latest_flare = get_cme_earth_impact = None

try:
    from anomaly_detector import check_all
except ImportError:
    check_all = None

try:
    from llm_briefing import generate_briefing, log_briefing
except ImportError:
    generate_briefing = log_briefing = None

OUT_FILE = "live_data.csv"
MODEL_FILE = "model_weights.json"
ORBIT_TYPE = "LEO"  # change to MEO / GEO if this collector is tracking a different orbit class


def get_recent_kp_history(n=2):
    """
    Reads the last n Kp values already logged in live_data.csv so the
    ML model has kp_1_ago / kp_2_ago without needing a separate feed.
    Returns a list, most recent last. Empty list if not enough history.
    """
    if not os.path.exists(OUT_FILE):
        return []
    try:
        with open(OUT_FILE, "r") as f:
            rows = list(csv.DictReader(f))
        vals = [float(r["kp"]) for r in rows[-n:] if r.get("kp") not in (None, "")]
        return vals
    except Exception as e:
        print(f"  -> could not read Kp history for ML model: {e}", flush=True)
        return []


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

    mag = fetch("https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json")
    if mag and len(mag) > 0:
        try:
            active_rows = [r for r in mag if r.get("active") is True]
            latest = active_rows[0] if active_rows else mag[0]
            bz = float(latest.get("bz_gsm", 0.0) or 0.0)
            bt = float(latest.get("bt", 0.0) or 0.0)
            print(f"  -> using mag row: time={latest.get('time_tag')} source={latest.get('source')} bz={bz} bt={bt}", flush=True)
        except Exception as e:
            print(f"  -> mag parse failed: {e}", flush=True)
    else:
        print("  -> rtsw_mag_1m.json failed/empty, bz/bt staying at 0.0", flush=True)

    plasma = fetch("https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json")
    if plasma and len(plasma) > 0:
        try:
            active_rows = [r for r in plasma if r.get("active") is True and r.get("proton_speed") is not None]
            latest = active_rows[0] if active_rows else None
            if latest:
                speed = float(latest.get("proton_speed", 0.0) or 0.0)
                density = float(latest.get("proton_density", 0.0) or 0.0)
                print(f"  -> using plasma row: time={latest.get('time_tag')} source={latest.get('source')} speed={speed} density={density}", flush=True)
            else:
                print("  -> no active plasma row with data found, speed/density staying at 0.0", flush=True)
        except Exception as e:
            print(f"  -> plasma parse failed: {e}", flush=True)
    else:
        print("  -> rtsw_wind_1m.json failed/empty, speed/density staying at 0.0", flush=True)

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

    # --- ML storm probability (real, trained model, separate from rule-based score) ---
    ml_prob, ml_alert = None, None
    if StormPredictor and os.path.exists(MODEL_FILE):
        history = get_recent_kp_history(n=2)
        if len(history) == 2:
            kp_1_ago, kp_2_ago = history[-1], history[-2]
            try:
                sp = StormPredictor(MODEL_FILE)
                ml_prob, ml_alert_bool, thresh = sp.classify(
                    d["current_kp"], kp_1_ago, kp_2_ago, sensitivity="balanced"
                )
                ml_alert = "YES" if ml_alert_bool else "no"
                print(f"  -> ML storm probability (next ~6h): {ml_prob:.1%}  alert={ml_alert} (threshold={thresh})", flush=True)
            except Exception as e:
                print(f"  -> ML scoring failed: {e}", flush=True)
        else:
            print(f"  -> not enough Kp history yet for ML model ({len(history)}/2 rows), skipping", flush=True)
    else:
        print("  -> model_weights.json or predict_storm.py not found, skipping ML scoring", flush=True)

    # --- Solar flare classification (real GOES data) ---
    flare_class = "unavailable"
    if get_latest_flare:
        try:
            flare_class, _ = get_latest_flare()
        except Exception as e:
            print(f"  -> flare check failed: {e}", flush=True)

    # --- CME Earth-impact check (real NASA DONKI data) ---
    cme_summary = "none detected"
    if get_cme_earth_impact:
        try:
            cme = get_cme_earth_impact()
            if cme:
                cme_summary = (f"speed={cme['cme_speed_km_s']}km/s, "
                                f"arrival in {cme['hours_until_arrival']}h, "
                                f"predicted Kp={cme['predicted_kp']}")
        except Exception as e:
            print(f"  -> CME check failed: {e}", flush=True)

    # --- Statistical anomaly check against this collector's own history ---
    anomaly_summary = "none"
    if check_all:
        try:
            anomalies = check_all({
                "kp": d["current_kp"], "bz": d["bz"],
                "solar_wind_speed": d["speed"], "density": d["density"]
            })
            flagged = [f"{field} (z={r['z_score']})" for field, r in anomalies.items() if r.get("is_anomaly")]
            anomaly_summary = ", ".join(flagged) if flagged else "none"
        except Exception as e:
            print(f"  -> anomaly check failed: {e}", flush=True)

    print("-" * 60, flush=True)
    print(f"FINAL VALUES: bz={d['bz']} bt={d['bt']} speed={d['speed']} density={d['density']} kp={d['current_kp']}", flush=True)
    print(f"Flare class: {flare_class} | CME: {cme_summary} | Anomalies: {anomaly_summary}", flush=True)
    print("-" * 60, flush=True)

    # --- LLM plain-English briefing (only runs if ANTHROPIC_API_KEY is set) ---
    if generate_briefing:
        try:
            briefing_input = {
                "kp": d["current_kp"], "bz": d["bz"], "speed": d["speed"], "density": d["density"],
                "risk_score": score, "risk_level": level,
                "ml_storm_probability": f"{ml_prob:.1%}" if ml_prob is not None else "unavailable",
                "flare_class": flare_class, "cme_info": cme_summary, "anomalies": anomaly_summary,
            }
            briefing_text, err = generate_briefing(briefing_input, orbit_type=ORBIT_TYPE)
            if err:
                print(f"  -> LLM briefing skipped: {err}", flush=True)
            else:
                print(f"  -> LLM briefing: {briefing_text}", flush=True)
                log_briefing(briefing_text, briefing_input)
        except Exception as e:
            print(f"  -> LLM briefing failed: {e}", flush=True)

    file_exists = os.path.exists(OUT_FILE)
    with open(OUT_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "kp", "bz", "bt", "solar_wind_speed",
                        "density", "max_kp_forecast", "ssn", "f107",
                        "active_alerts", "risk_score", "risk_level",
                        "ml_storm_probability", "ml_alert",
                        "flare_class", "cme_alert", "anomaly_flag"])
        w.writerow([now_str, d["current_kp"], d["bz"], d["bt"], d["speed"],
                    d["density"], d["max_kp"], d["ssn"], d["f107"],
                    d["alert_count"], score, level,
                    f"{ml_prob:.4f}" if ml_prob is not None else "",
                    ml_alert if ml_alert is not None else "",
                    flare_class,
                    cme_summary if cme_summary != "none detected" else "no",
                    anomaly_summary])

    print(f"{now_str} | Kp={d['current_kp']} | risk={score}/10 ({level}) | ml_prob={ml_prob:.1%}" if ml_prob is not None
          else f"{now_str} | Kp={d['current_kp']} | risk={score}/10 ({level}) | ml_prob=n/a", flush=True)


if __name__ == "__main__":
    main()
