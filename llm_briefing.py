"""
llm_briefing.py
------------------------------------------------------------
Fuses rule-based risk score, ML storm probability, flare class,
CME status, and anomaly flags into a short plain-English
briefing an operator can read in 10 seconds, personalized per
orbit type. Uses the Anthropic API (Claude Haiku -- cheap and
fast, the right size model for a job run every 2 hours).

Cost: at Haiku-class pricing, 12 briefings/day (one per 2-hour
collector run) costs a small fraction of a typical monthly
budget -- comfortably inside a 1500-2000 INR/month ceiling with
large headroom for other usage too.

Needs an ANTHROPIC_API_KEY environment variable, added as a
GitHub Actions secret (Settings -> Secrets and variables ->
Actions -> New repository secret), then referenced in the
workflow yml as:
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
------------------------------------------------------------
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"
LOG_FILE = "briefings_log.jsonl"


def _call_claude(system_prompt, user_prompt, api_key, max_tokens=300):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            parts = [b["text"] for b in resp.get("content", []) if b.get("type") == "text"]
            return "\n".join(parts).strip(), None
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")[:300]
        return None, f"HTTP {e.code}: {err_body}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def generate_briefing(briefing_input, orbit_type="LEO", api_key=None):
    """
    briefing_input: flat dict of current readings + model outputs
                    (kp, bz, speed, density, risk_score, risk_level,
                     ml_storm_probability, flare_class, cme_info, anomalies)
    orbit_type: "LEO", "MEO", or "GEO"

    Returns (briefing_text, error_message). Exactly one of the two
    will be None -- callers should check `err` first.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set"

    system_prompt = (
        "You are a space weather risk analyst writing a short operational "
        "briefing for a satellite operator. Be concrete and honest -- never "
        "overstate confidence. If risk is low, say so plainly and briefly. "
        "Ground every claim in the numbers given; do not invent data. "
        "Write 3-5 sentences, plain English, no headers, no bullet points. "
        "Mention the specific orbit-relevant risk (atmospheric drag for LEO, "
        "radiation belt exposure for MEO, surface charging/single-event "
        "upsets for GEO) only if the data suggests it's actually relevant "
        "right now -- do not force it in if conditions are quiet."
    )
    user_prompt = (
        f"Orbit type: {orbit_type}\n\n"
        f"Current readings and model outputs (JSON):\n{json.dumps(briefing_input, indent=2)}\n\n"
        "Write the operator briefing now."
    )
    return _call_claude(system_prompt, user_prompt, api_key)


def log_briefing(briefing_text, briefing_input, log_path=LOG_FILE):
    """Appends the briefing + the data that produced it to a JSONL log,
    so you have a record for the pitch deck / demo without re-calling the API."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "briefing": briefing_text,
        "input": briefing_input,
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  -> could not write briefing log: {e}", flush=True)


if __name__ == "__main__":
    sample = {
        "kp": 6.3, "bz": -12.5, "speed": 680, "density": 4.1,
        "risk_score": 7, "risk_level": "HIGH",
        "ml_storm_probability": "64.0%",
        "flare_class": "M1.2",
        "cme_info": "speed=850km/s, arrival in 14.5h, predicted Kp=6",
        "anomalies": "bz (z=-3.1)",
    }
    text, err = generate_briefing(sample, orbit_type="LEO")
    if err:
        print(f"Error: {err}")
    else:
        print(text)
        log_briefing(text, sample)
