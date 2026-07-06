"""
fetch_space_weather_plus.py
------------------------------------------------------------
Adds two real data sources beyond Kp/solar wind:

1. Solar X-ray flares (NOAA GOES) -- classifies current flux
   into the standard A/B/C/M/X flare scale.
2. Coronal Mass Ejections (NASA DONKI) -- checks for any CME
   with an Earth-directed impact prediction (isEarthGB or
   isEarthMinorImpact) and returns the soonest predicted arrival.

Both are free, keyless (DONKI works with DEMO_KEY, though a
free personal key from api.nasa.gov raises the rate limit and
costs nothing -- recommended for the live collector).
------------------------------------------------------------
"""

import urllib.request
import urllib.error
import json
import math
from datetime import datetime, timedelta, timezone

NASA_API_KEY = "DEMO_KEY"  # replace with a free key from https://api.nasa.gov if you hit rate limits

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def _fetch(url):
    try:
        req = urllib.request.Request(url, headers=BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  -> fetch failed for {url}: {e}", flush=True)
        return None


def classify_flare(flux):
    """Standard NOAA GOES X-ray flare classification from long-band (0.1-0.8nm) flux in W/m^2."""
    if flux is None or flux <= 0:
        return "Unknown", 0.0
    if flux < 1e-7:
        return "A (background)", flux
    exponent = math.floor(math.log10(flux))
    coefficient = flux / (10 ** exponent)
    scale_map = {-7: "B", -6: "C", -5: "M", -4: "X"}
    letter = scale_map.get(exponent, "X" if exponent > -4 else "A")
    return f"{letter}{coefficient:.1f}", flux


def get_latest_flare():
    """Returns (flare_class_str, raw_flux) from the most recent long-band GOES reading."""
    data = _fetch("https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json")
    if not data:
        return "Unknown", None
    long_band = [r for r in data if r.get("energy") == "0.1-0.8nm" and r.get("flux") is not None]
    if not long_band:
        return "Unknown", None
    long_band.sort(key=lambda r: r["time_tag"])
    latest = long_band[-1]
    flare_class, flux = classify_flare(latest["flux"])
    return flare_class, flux


def get_cme_earth_impact(days_back=5):
    """
    Checks NASA DONKI for CMEs in the last `days_back` days with a
    predicted Earth impact (via the WSA-ENLIL model). Returns the
    soonest predicted arrival, or None if no Earth-directed CME found.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    url = (
        f"https://api.nasa.gov/DONKI/CME?startDate={start.strftime('%Y-%m-%d')}"
        f"&endDate={end.strftime('%Y-%m-%d')}&api_key={NASA_API_KEY}"
    )
    data = _fetch(url)
    if not data:
        return None

    candidates = []
    for cme in data:
        for analysis in cme.get("cmeAnalyses") or []:
            for enlil in analysis.get("enlilList") or []:
                if enlil.get("isEarthGB") or enlil.get("isEarthMinorImpact"):
                    arrival = enlil.get("estimatedShockArrivalTime")
                    if arrival:
                        try:
                            arrival_dt = datetime.fromisoformat(arrival.replace("Z", "+00:00"))
                            candidates.append({
                                "activity_id": cme.get("activityID"),
                                "cme_speed_km_s": analysis.get("speed"),
                                "estimated_arrival": arrival,
                                "hours_until_arrival": round((arrival_dt - end).total_seconds() / 3600, 1),
                                "predicted_kp": enlil.get("kp_180") or enlil.get("kp_135") or enlil.get("kp_90"),
                                "geomagnetic_storm_expected": bool(enlil.get("isEarthGB")),
                            })
                        except Exception:
                            continue

    if not candidates:
        return None

    # Return the soonest upcoming arrival (could be in the past if already arrived recently)
    candidates.sort(key=lambda c: c["hours_until_arrival"])
    upcoming = [c for c in candidates if c["hours_until_arrival"] >= -12]  # include very recent arrivals
    return upcoming[0] if upcoming else candidates[0]


if __name__ == "__main__":
    print("Checking latest solar flare class...")
    flare_class, flux = get_latest_flare()
    print(f"  Flare class: {flare_class}  (flux={flux})")

    print("\nChecking for Earth-directed CMEs (last 5 days)...")
    cme = get_cme_earth_impact()
    if cme:
        print(f"  CME found: {json.dumps(cme, indent=2)}")
    else:
        print("  No Earth-directed CME with a predicted impact in this window.")
