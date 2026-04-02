import argparse
import sys
import os
import time
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

def smart_agent_action(state, step_num, prev_zone_survivors):
    zones = state.get("zones", [])
    
    # Step 1 - Calculate triage score per zone
    active_zones = []
    for z in zones:
        if z.get("contained", False) or z.get("survivors", 0) == 0:
            continue
        try:
            triage = z["severity"] * z["survivors"] * (2.0 if z.get("time_critical") else 1.0) * z["accessibility"]
            active_zones.append((triage, z))
        except KeyError:
            active_zones.append((0.0, z))
            
    # sort zones by triage descending
    active_zones.sort(key=lambda x: x[0], reverse=True)
    sorted_zones = [z for _, z in active_zones]

    if not sorted_zones:
        return {"deployments": []}

    # Step 2 - Calculate budget for this step
    steps_left = state.get("time_remaining", 1)
    spend_ratio = min(0.45, 1.0 / max(1, steps_left))
    # this naturally increases spending as time runs out
    
    res = state.get("resources", {})
    budget_ambu = max(1, int(res.get("ambulances", 0) * spend_ratio)) if res.get("ambulances", 0) > 0 else 0
    budget_teams = max(1, int(res.get("rescue_teams", 0) * spend_ratio)) if res.get("rescue_teams", 0) > 0 else 0
    budget_food = max(1, int(res.get("food_packets", 0) * spend_ratio)) if res.get("food_packets", 0) > 0 else 0

    # Step 4 - Handle Zone E reveal (task_hard)
    override_zone = None
    for z in sorted_zones:
        zid = z["zone_id"]
        # If any zone has survivors > 0 AND was previously survivors == 0
        if z["survivors"] > 0 and prev_zone_survivors.get(zid, -1) == 0:
            override_zone = z
            break
            
    deployments = {}
    
    a_avail = res.get("ambulances", 0)
    r_avail = res.get("rescue_teams", 0)
    f_avail = res.get("food_packets", 0)

    if override_zone:
        zid = override_zone["zone_id"]
        a_dep = min(a_avail, max(1, int(budget_ambu * 0.5)))
        r_dep = min(r_avail, max(1, int(budget_teams * 0.5)))
        f_dep = min(f_avail, max(1, int(budget_food * 0.5)))
        
        needed = override_zone.get("resources_needed", {})
        a_dep = min(a_dep, needed.get("ambulances", 0))
        r_dep = min(r_dep, needed.get("rescue_teams", 0))
        f_dep = min(f_dep, needed.get("food_packets", 0))

        if override_zone.get("time_critical"):
            a_dep = max(1 if a_avail > 0 else 0, a_dep)
            r_dep = max(1 if r_avail > 0 else 0, r_dep)
            f_dep = max(1 if f_avail > 0 else 0, f_dep)

        deployments[zid] = {
            "ambulances": a_dep,
            "rescue_teams": r_dep,
            "food_packets": f_dep,
            "priority": 1 # priority=1 override
        }
        
        sorted_zones = [z for z in sorted_zones if z["zone_id"] != zid]
        budget_ambu -= a_dep
        budget_teams -= r_dep
        budget_food -= f_dep
        a_avail -= a_dep
        r_avail -= r_dep
        f_avail -= f_dep

    # Step 3 - Distribute budget across top zones
    shares = [0.60, 0.30, 0.10]
    for i, z in enumerate(sorted_zones[:3]):
        zid = z["zone_id"]
        share = shares[i]
        
        a_share = int(budget_ambu * share)
        r_share = int(budget_teams * share)
        f_share = int(budget_food * share)
        
        needed = z.get("resources_needed", {})
        
        a_dep = min(needed.get("ambulances", 0), a_share)
        r_dep = min(needed.get("rescue_teams", 0), r_share)
        f_dep = min(needed.get("food_packets", 0), f_share)
        
        # Never send 0 to a time_critical zone — minimum 1 of each resource
        if z.get("time_critical"):
            a_dep = max(1 if a_avail > 0 else 0, a_dep)
            r_dep = max(1 if r_avail > 0 else 0, r_dep)
            f_dep = max(1 if f_avail > 0 else 0, f_dep)

        # Assure we don't exceed what is available overall
        a_dep = min(a_dep, a_avail)
        r_dep = min(r_dep, r_avail)
        f_dep = min(f_dep, f_avail)
        
        deployments[zid] = {
            "ambulances": a_dep,
            "rescue_teams": r_dep,
            "food_packets": f_dep,
            "priority": 1 if z.get("time_critical") else 3
        }
        
        a_avail -= a_dep
        r_avail -= r_dep
        f_avail -= f_dep

    # Step 5 - Build Action dict
    deps_list = []
    for zid, d in deployments.items():
        # Only include zones where at least one resource > 0
        if d["ambulances"] > 0 or d["rescue_teams"] > 0 or d["food_packets"] > 0:
            deps_list.append({
                "zone_id": zid,
                "ambulances": d["ambulances"],
                "rescue_teams": d["rescue_teams"],
                "food_packets": d["food_packets"],
                "priority": d["priority"]
            })
            
    return {"deployments": deps_list}

def llm_agent_action(state, step_num):
    # Dummy fallback path when passing --agent llm
    print("LLM path called, but not fully implemented.")
    return {"deployments": []}

def run_task(task_id: str, base_url: str, agent_type: str) -> float:
    url = base_url.rstrip("/")
    with httpx.Client(timeout=30.0) as http:
        for attempt in range(3):
            try:
                resp = http.post(f"{url}/reset", json={"task_id": task_id, "seed": 42})
                resp.raise_for_status()
                break
            except Exception as e:
                time.sleep(1)

        print(f"\nRunning {task_id} with agent={agent_type}...")
        step_num = 0
        prev_zones = {}

        while True:
            for attempt in range(3):
                try:
                    resp = http.get(f"{url}/state")
                    resp.raise_for_status()
                    state_data = resp.json()
                    break
                except Exception as e:
                    time.sleep(1)
            else:
                print("Failed to get state.")
                break

            if state_data.get("done"):
                return state_data.get("score", 0.0)

            state = state_data.get("state", {})
            step_num += 1

            if step_num == 1:
                # Initialize previous tracking for step 1
                prev_zones = {z["zone_id"]: z.get("survivors", 0) for z in state.get("zones", [])}

            if agent_type == "rule":
                action = smart_agent_action(state, step_num, prev_zones)
            else:
                action = llm_agent_action(state, step_num)

            for attempt in range(3):
                try:
                    resp = http.post(f"{url}/step", json=action)
                    resp.raise_for_status()
                    step_resp = resp.json()

                    score = step_resp.get("score", 0.0)
                    new_st = step_resp.get("state", {})
                    
                    # Track prev_zone_survivors after each step response
                    prev_zones = {z["zone_id"]: z.get("survivors", 0) for z in new_st.get("zones", [])}

                    deps = "+".join(f"{d['zone_id']}(A{d['ambulances']})" for d in action.get("deployments", []))
                    sv = " | ".join(f"{z['zone_id']}:{z['survivors']}" for z in new_st.get("zones", []) if z.get("survivors",0)>0)
                    print(f"  Step {step_num:2d} | Score: {score:.4f} | A={new_st.get('resources',{}).get('ambulances',0)} -> [{deps}] | Surv: [{sv}]")

                    if step_resp.get("done"):
                        return score
                    break
                except Exception as e:
                    time.sleep(1)
            else:
                return score
            time.sleep(0.05)
    return 0.0

def main():
    parser = argparse.ArgumentParser(description="CrisisFlow Agent Inference")
    parser.add_argument("--agent", choices=["rule", "llm"], default="rule", help="Choose between rule or llm agent")
    args = parser.parse_args()

    scores = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        score = run_task(task_id, API_BASE_URL, args.agent)
        scores[task_id] = score

    print("\n" + "=" * 50)
    for t in scores:
        print(f"  {t:15s}: {scores[t]:.4f}")
    
    thresholds = {"task_easy": 0.75, "task_medium": 0.70, "task_hard": 0.60}
    all_pass = True
    for tid, thresh in thresholds.items():
        if scores[tid] < thresh:
            all_pass = False

    print(f"\n  Validation passing: {'YES' if all_pass else 'NO'}")
    print("=" * 50)

if __name__ == "__main__":
    main()
