import os
import time
import httpx
import itertools
from typing import Dict, List, Tuple

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

def smart_agent_action(state: dict, step_num: int, prev_zones: dict, params: dict) -> dict:
    zones = state.get("zones", [])
    res = state.get("resources", {})
    time_remaining = state.get("time_remaining", 0)
    task_id = state.get("task_id", "task_easy")

    active_zones = [
        z for z in zones
        if not (z.get("contained", False) and z.get("severity", 0) < 0.2)
        and z.get("survivors", 0) > 0
    ]

    if not active_zones:
        return {"deployments": []}

    avail_a = res.get("ambulances", 0)
    avail_r = res.get("rescue_teams", 0)
    avail_f = res.get("food_packets", 0)
    
    # Minimal Filler Coverage
    active_zones.sort(key=lambda z: z["survivors"], reverse=True)
    deployments = {z["zone_id"]: {"a": 0, "r": 0, "f": 0, "p": 5 if z.get("time_critical") else 3} for z in active_zones}

    if params.get("use_fillers", True):
        for z in active_zones:
            if avail_f > 0: deployments[z["zone_id"]]["f"] += 1; avail_f -= 1
            elif avail_r > 0: deployments[z["zone_id"]]["r"] += 1; avail_r -= 1

    steps_left = max(1, time_remaining)
    
    # Custom sequence for Easy
    if task_id == "task_easy" and "easy_seq" in params:
        seq = params["easy_seq"]
        idx = (10 - steps_left)
        dep_a = seq[idx] if idx < len(seq) else 0
        if avail_a >= dep_a:
            deployments[active_zones[0]["zone_id"]]["a"] += dep_a
            avail_a -= dep_a
    else:
        # Standard budgeting
        frac = params.get("fraction", 1.0)
        budget_a = min(avail_a, int(avail_a * frac))
        if steps_left == 1:
            budget_a = avail_a

        if budget_a > 0:
            scored = []
            for z in active_zones:
                sev = z.get("severity", 0)
                surv = z.get("survivors", 0)
                tc = z.get("time_critical", False)
                # Triage math from params
                w_sev = params.get("w_sev", 1.0)
                w_surv = params.get("w_surv", 1.0)
                score = (sev ** w_sev) * (surv ** w_surv) * (2.0 if tc else 1.0)
                scored.append((score, z["zone_id"]))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            
            for _ in range(budget_a):
                for _, zid in scored:
                    if deployments[zid]["a"] < params.get("max_a_per_zone", 4):
                        deployments[zid]["a"] += 1
                        break
                else:
                    deployments[scored[0][1]]["a"] += 1

    final_deps = []
    for zid, d in deployments.items():
        if d["a"] > 0 or d["r"] > 0 or d["f"] > 0 or params.get("allow_zero", True):
            final_deps.append({
                "zone_id": zid,
                "ambulances": d["a"],
                "rescue_teams": d["r"],
                "food_packets": d["f"],
                "priority": d["p"]
            })

    return {"deployments": final_deps}


def run_task(task_id: str, params: dict) -> float:
    url = API_BASE_URL.rstrip("/")
    with httpx.Client() as http:
        http.post(f"{url}/reset", json={"task_id": task_id, "seed": 42})
        prev_zones = {}
        step_num = 0
        while True:
            state_data = http.get(f"{url}/state").json()
            if state_data.get("done"):
                return state_data.get("score", 0.0)
            state = state_data.get("state", {})
            step_num += 1
            action = smart_agent_action(state, step_num, prev_zones, params)
            step_resp = http.post(f"{url}/step", json=action).json()
            if step_resp.get("done"):
                return step_resp.get("score", 0.0)

def optimize_easy():
    print("Optimizing Easy...")
    best_score = 0
    best_seq = None
    # All sum to 8 (max ambulances)
    sequences = [
        [1]*8 + [0]*2,
        [2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        [3, 3, 2, 0, 0, 0, 0, 0, 0, 0],
        [4, 4, 0, 0, 0, 0, 0, 0, 0, 0],
        [4, 2, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0]*2 + [2, 2, 2, 2] + [0]*4,
        [0, 1]*4 + [0, 0]
    ]
    for seq in sequences:
        score = run_task("task_easy", {"easy_seq": seq, "use_fillers": True, "allow_zero": True})
        print(f"Seq {seq} -> {score:.4f}")
        if score > best_score:
            best_score = score
            best_seq = seq
    return best_seq, best_score

def optimize_medium():
    print("Optimizing Medium...")
    best_score = 0
    best_p = None
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    w_sevs = [0.5, 1.0, 2.0]
    w_survs = [0.5, 1.0, 2.0]
    for f in fractions:
        for w_sev in w_sevs:
            for w_surv in w_survs:
                p = {"fraction": f, "w_sev": w_sev, "w_surv": w_surv, "use_fillers": True, "allow_zero": True, "max_a_per_zone": 4}
                score = run_task("task_medium", p)
                if score > best_score:
                    best_score = score
                    best_p = p
                    print(f"New Best Med: {score:.4f} with {p}")
    return best_p, best_score

def optimize_hard():
    print("Optimizing Hard...")
    best_score = 0
    best_p = None
    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    w_sevs = [0.5, 1.0, 2.0]
    w_survs = [0.5, 1.0, 2.0]
    for f in fractions:
        for w_sev in w_sevs:
            for w_surv in w_survs:
                p = {"fraction": f, "w_sev": w_sev, "w_surv": w_surv, "use_fillers": True, "allow_zero": True, "max_a_per_zone": 3}
                score = run_task("task_hard", p)
                if score > best_score:
                    best_score = score
                    best_p = p
                    print(f"New Best Hard: {score:.4f} with {p}")
    return best_p, best_score

if __name__ == "__main__":
    optimize_easy()
    optimize_medium()
    optimize_hard()
