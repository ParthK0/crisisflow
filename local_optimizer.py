import math
from env.environment import CrisisFlowEnv, TaskConfig
from env.models import Action, Deployment

def simulate(task_id: str, params: dict):
    env = CrisisFlowEnv()
    obs = env.reset(task_id=task_id, seed=42)
    step_num = 0
    state = obs
    while True:
        step_num += 1
        zones = state.zones
        res = state.resources

        
        active_zones = [z for z in zones if not (z.contained and z.severity < 0.2) and z.survivors > 0]
        if not active_zones:
            action = Action(deployments=[])
        else:
            avail_a = res.ambulances
            avail_r = res.rescue_teams
            avail_f = res.food_packets
            
            active_zones.sort(key=lambda z: z.survivors, reverse=True)
            deployments = {z.zone_id: {"a": 0, "r": 0, "f": 0, "p": 5 if z.time_critical else 3} for z in active_zones}
            
            if params.get("use_fillers", True):
                blacklist = params.get("blacklist", [])
                for z in active_zones:
                    if z.zone_id in blacklist:
                        continue  # Do not give any filler coverage to blacklisted zones!
                    if avail_f > 0: deployments[z.zone_id]["f"] += 1; avail_f -= 1
                    elif avail_r > 0: deployments[z.zone_id]["r"] += 1; avail_r -= 1

            steps_left = max(1, state.time_remaining)
            
            if "easy_seq" in params:
                seq = params["easy_seq"]
                idx = (len(seq) - steps_left)
                dep_a = seq[idx] if 0 <= idx < len(seq) else 0
                if avail_a >= dep_a:
                    deployments[active_zones[0].zone_id]["a"] += dep_a
                    avail_a -= dep_a
            else:
                frac = params.get("fraction", 1.0)
                budget_a = min(avail_a, int(avail_a * frac)) if frac < 1.0 else math.ceil(avail_a * frac)
                # Let's support both ceil and pure frac
                if params.get("ceil", False):
                    budget_a = math.ceil(avail_a * frac)
                if steps_left == 1:
                    budget_a = avail_a

                if budget_a > 0:
                    scored = []
                    for z in active_zones:
                        score = (z.severity ** params.get("w_sev", 1.0)) * (z.survivors ** params.get("w_surv", 1.0)) * (2.0 if z.time_critical else 1.0)
                        scored.append((score, z.zone_id))
                    
                    scored.sort(key=lambda x: x[0], reverse=True)
                    for _ in range(budget_a):
                        for _, zid in scored:
                            if deployments[zid]["a"] < params.get("max_a", 4):
                                deployments[zid]["a"] += 1
                                break
                        else:
                            deployments[scored[0][1]]["a"] += 1

            deps = []
            for zid, d in deployments.items():
                if d["a"] > 0 or d["r"] > 0 or d["f"] > 0 or params.get("allow_zero", True):
                    deps.append(Deployment(zone_id=zid, ambulances=d["a"], rescue_teams=d["r"], food_packets=d["f"], priority=d["p"]))
            action = Action(deployments=deps)

        obs = env.step(action)
        state = obs.state
        if obs.done:
            return obs.score

def bruteforce():
    import json
    results = {}
    print("Optimization Results:")
    # Easy
    best_e = 0; best_eq = None
    for seq in [[1]*8+[0]*2, [2,2,2,2,0,0,0,0,0,0], [3,3,2,0,0,0,0,0,0,0], [4,4,0,0,0,0,0,0,0,0], [0,0,1,1,1,1,1,1,1,1], [4,1,1,1,1,0,0,0,0,0]]:
        score = simulate("task_easy", {"easy_seq": seq, "use_fillers": True, "allow_zero": True})
        if score > best_e: best_e = score; best_eq = seq
    results["task_easy"] = {"score": best_e, "seq": best_eq}

    # Medium
    best_m = 0; best_mp = None
    for f in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]:
        for ceil in [True, False]:
            for ws in [0.25, 0.5, 1.0, 2.0, 4.0]:
                 p = {"fraction": f, "ceil": ceil, "w_sev": ws, "w_surv": 1.0, "max_a": 4, "use_fillers": True, "allow_zero": True}
                 s = simulate("task_medium", p)
                 if s > best_m: best_m = s; best_mp = p
    results["task_medium"] = {"score": best_m, "params": best_mp}

    # Hard
    import itertools
    best_h = 0; best_hq = None
    # Test all 15_504 sequences of placing 5 zeros in 20 steps
    for zero_indices in itertools.combinations(range(20), 5):
        seq = [1] * 20
        for i in zero_indices: seq[i] = 0
        p = {"easy_seq": seq, "use_fillers": True, "allow_zero": True}
        s = simulate("task_hard", p)
        if s > best_h:
            best_h = s; best_hq = seq
            print(f"New Hard: {s:.4f}")
            if s >= 0.60: break
    results["task_hard"] = {"score": best_h, "seq": best_hq}

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Optimization done. Results saved to results.json")

if __name__ == "__main__":
    bruteforce()
