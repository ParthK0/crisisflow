#!/usr/bin/env python3
"""Pre-submission checker: HTTP checks matching judge automation (httpx)."""

from __future__ import annotations

import os
import random
import sys
from typing import Any, Callable, List, Tuple

import httpx

DEFAULT_BASE = "http://localhost:7860"
BASE_URL = os.environ.get("BASE_URL", DEFAULT_BASE).rstrip("/")

CheckFn = Callable[[httpx.Client], Tuple[bool, str]]

RESULTS: List[Tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"{tag} - {name}"
    if detail:
        line += f" - {detail}"
    print(line)
    RESULTS.append((name, ok, detail))


def check_health(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.get("/health", timeout=10.0)
        if r.status_code != 200:
            return False, f"status {r.status_code}"
        data = r.json()
        if data.get("status") != "ok":
            return False, f"body missing status ok: {data!r}"
        return True, ""
    except Exception as e:
        return False, str(e)


def check_tasks(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.get("/tasks", timeout=10.0)
        if r.status_code != 200:
            return False, f"status {r.status_code}"
        data = r.json()
        if not isinstance(data, list):
            return False, "not a list"
        if len(data) != 3:
            return False, f"expected 3 items, got {len(data)}"
        need_fields = ("id", "name", "difficulty", "max_steps", "pass_threshold")
        diffs: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                return False, "item not an object"
            for f in need_fields:
                if f not in item:
                    return False, f"missing field {f!r}"
            d = item.get("difficulty")
            if isinstance(d, str):
                diffs.append(d)
        if diffs != ["easy", "medium", "hard"]:
            return False, f"difficulties order {diffs!r}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _disaster_state_ok(d: Any) -> Tuple[bool, str]:
    if not isinstance(d, dict):
        return False, "not an object"
    for k in ("zones", "resources", "time_remaining", "step_count", "cumulative_reward", "task_id", "seed"):
        if k not in d:
            return False, f"missing {k!r}"
    if not isinstance(d.get("zones"), list):
        return False, "zones not a list"
    if not isinstance(d.get("resources"), dict):
        return False, "resources not an object"
    return True, ""


def check_reset_easy(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.post("/reset", json={"task_id": "task_easy", "seed": 42}, timeout=10.0)
        if r.status_code != 200:
            return False, f"status {r.status_code}"
        data = r.json()
        ok, err = _disaster_state_ok(data)
        if not ok:
            return False, err
        if len(data["zones"]) != 1:
            return False, f"zones len {len(data['zones'])}"
        return True, ""
    except Exception as e:
        return False, str(e)


def check_state_after_reset(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.get("/state", timeout=10.0)
        if r.status_code != 200:
            return False, f"status {r.status_code}"
        data = r.json()
        ok, err = _disaster_state_ok(data)
        if not ok:
            return False, err
        return True, ""
    except Exception as e:
        return False, str(e)


def check_step_valid(client: httpx.Client) -> Tuple[bool, str]:
    try:
        zid = "z1"
        r = client.post(
            "/step",
            json={
                "deployments": [
                    {
                        "zone_id": zid,
                        "ambulances": 1,
                        "rescue_teams": 0,
                        "food_packets": 0,
                        "priority": 3,
                    }
                ]
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return False, f"status {r.status_code} {r.text[:200]}"
        data = r.json()
        for k in ("state", "reward", "done", "score", "info"):
            if k not in data:
                return False, f"missing {k!r}"
        if not isinstance(data["reward"], (int, float)):
            return False, "reward not numeric"
        if not isinstance(data["done"], bool):
            return False, "done not bool"
        return True, ""
    except Exception as e:
        return False, str(e)


def check_step_overdeploy(client: httpx.Client) -> Tuple[bool, str]:
    try:
        client.post("/reset", json={"task_id": "task_easy", "seed": 99}, timeout=10.0)
        r = client.post(
            "/step",
            json={
                "deployments": [
                    {
                        "zone_id": "z1",
                        "ambulances": 9999,
                        "rescue_teams": 0,
                        "food_packets": 0,
                        "priority": 3,
                    }
                ]
            },
            timeout=10.0,
        )
        if 400 <= r.status_code < 500:
            return True, ""
        return False, f"expected 4xx, got {r.status_code}"
    except Exception as e:
        return False, str(e)


def check_reset_medium(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.post("/reset", json={"task_id": "task_medium", "seed": 1}, timeout=10.0)
        if r.status_code != 200:
            return False, f"status {r.status_code}"
        data = r.json()
        if len(data.get("zones") or []) != 3:
            return False, f"zones len {len(data.get('zones') or [])}"
        return True, ""
    except Exception as e:
        return False, str(e)


def check_reset_hard(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.post("/reset", json={"task_id": "task_hard", "seed": 1}, timeout=10.0)
        if r.status_code != 200:
            return False, f"status {r.status_code}"
        data = r.json()
        if len(data.get("zones") or []) != 5:
            return False, f"zones len {len(data.get('zones') or [])}"
        return True, ""
    except Exception as e:
        return False, str(e)


def check_unknown_task(client: httpx.Client) -> Tuple[bool, str]:
    try:
        r = client.post("/reset", json={"task_id": "nonexistent", "seed": 1}, timeout=10.0)
        if r.status_code in (400, 422):
            return True, ""
        return False, f"expected 400 or 422, got {r.status_code}"
    except Exception as e:
        return False, str(e)


def check_score_range_episode(client: httpx.Client) -> Tuple[bool, str]:
    try:
        rng = random.Random(12345)
        client.post("/reset", json={"task_id": "task_easy", "seed": 42}, timeout=10.0)
        final_score: float | None = None
        for _ in range(50):
            st = client.get("/state", timeout=10.0)
            if st.status_code != 200:
                return False, f"state {st.status_code}"
            body = st.json()
            pools = body["resources"]
            zid = body["zones"][0]["zone_id"]
            amb = rng.randint(0, min(3, pools["ambulances"]))
            rt = rng.randint(0, min(3, pools["rescue_teams"]))
            fd = rng.randint(0, min(3, pools["food_packets"]))
            sr = client.post(
                "/step",
                json={
                    "deployments": [
                        {
                            "zone_id": zid,
                            "ambulances": amb,
                            "rescue_teams": rt,
                            "food_packets": fd,
                            "priority": rng.randint(1, 5),
                        }
                    ]
                },
                timeout=10.0,
            )
            if sr.status_code != 200:
                return False, f"step {sr.status_code} {sr.text[:120]}"
            out = sr.json()
            final_score = float(out["score"])
            if out.get("done"):
                break
        if final_score is None:
            return False, "no score"
        if not (0.0 <= final_score <= 1.0):
            return False, f"score {final_score}"
        return True, f"final score={final_score:.4f}"
    except Exception as e:
        return False, str(e)


CHECKS: List[Tuple[str, CheckFn]] = [
    ("Health check", check_health),
    ("Tasks endpoint", check_tasks),
    ("Reset easy", check_reset_easy),
    ("State after reset", check_state_after_reset),
    ("Step with valid action", check_step_valid),
    ("Step over-deploy (4xx)", check_step_overdeploy),
    ("Reset medium", check_reset_medium),
    ("Reset hard", check_reset_hard),
    ("Unknown task", check_unknown_task),
    ("Score range (task_easy episode)", check_score_range_episode),
]


def main() -> int:
    print(f"BASE_URL={BASE_URL}\n")
    ok_count = 0
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        for name, fn in CHECKS:
            passed, detail = fn(client)
            _record(name, passed, detail)
            if passed:
                ok_count += 1

    total = len(CHECKS)
    fail_count = total - ok_count
    print()
    print(f"Passed: {ok_count}/{total}")
    print(f"Failed: {fail_count}/{total}")
    if fail_count == 0:
        print("Status: READY TO SUBMIT")
        return 0
    print("Status: FIX BEFORE SUBMITTING")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
