#!/usr/bin/env python3
"""Baseline agent: typed actions (deployments) against the local CrisisFlow API."""

from __future__ import annotations

import os
import random
import sys

import httpx

BASE = os.environ.get("CRISISFLOW_BASE_URL", "http://127.0.0.1:7860").rstrip("/")
TASK = os.environ.get("CRISISFLOW_TASK", "task_easy")
MAX_STEPS = int(os.environ.get("CRISISFLOW_MAX_STEPS", "25"))


def main() -> int:
    rng = random.Random(0)
    with httpx.Client(base_url=BASE, timeout=30.0) as client:
        r = client.post("/reset", json={"task_id": TASK, "seed": 42})
        r.raise_for_status()
        for _ in range(MAX_STEPS):
            st = client.get("/state")
            st.raise_for_status()
            data = st.json()
            if data.get("done"):
                break
            st_obj = data.get("state", {})
            zones = st_obj.get("zones", [])
            pool = st_obj.get("resources", {})
            
            zone_ids = [z["zone_id"] for z in zones] if zones else ["z1"]
            zid = rng.choice(zone_ids)
            
            # Smart deployment: don't exceed pool
            a_avail = pool.get("ambulances", 0)
            r_avail = pool.get("rescue_teams", 0)
            f_avail = pool.get("food_packets", 0)
            
            payload = {
                "deployments": [
                    {
                        "zone_id": zid,
                        "ambulances": rng.randint(0, min(2, a_avail)),
                        "rescue_teams": rng.randint(0, min(2, r_avail)),
                        "food_packets": rng.randint(0, min(2, f_avail)),
                        "priority": rng.randint(1, 5),
                    }
                ]
            }
            sr = client.post("/step", json=payload)
            sr.raise_for_status()
            body = sr.json()
            if body.get("done"):
                break
        final = client.get("/state").json()
        score = float(final.get("score", 0.0))
        print(f"episode cumulative score: {score:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.ConnectError:
        print(
            f"Cannot connect to {BASE}. Start the server: uvicorn server.app:app --port 7860",
            file=sys.stderr,
        )
        raise SystemExit(1)
