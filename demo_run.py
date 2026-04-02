"""Quick demo: run all 3 tasks with a simple heuristic agent (no LLM needed)."""
import httpx

BASE = "http://localhost:7860"

def run_demo():
    with httpx.Client(base_url=BASE, timeout=10) as c:
        # Health check
        h = c.get("/health").json()
        print(f"Server: {h['status']} (v{h['version']})")

        # List tasks
        tasks = c.get("/tasks").json()
        print(f"Available tasks: {[t['id'] for t in tasks]}\n")

        for task in tasks:
            tid = task["id"]
            print(f"{'='*50}")
            print(f"Running: {tid} ({task['name']}) - {task['difficulty']}")
            print(f"Max steps: {task['max_steps']} | Pass threshold: {task['pass_threshold']}")
            print(f"{'='*50}")

            # Reset
            r = c.post("/reset", json={"task_id": tid, "seed": 42})
            state_resp = r.json()
            st = state_resp["state"]
            print(f"Zones: {len(st['zones'])} | Resources: A={st['resources']['ambulances']} R={st['resources']['rescue_teams']} F={st['resources']['food_packets']}")

            step_num = 0
            while True:
                # Check state
                state_resp = c.get("/state").json()
                if state_resp.get("done"):
                    print(f"  >> Episode complete! Final score: {state_resp['score']:.4f}")
                    break

                st = state_resp["state"]
                res = st["resources"]

                # Simple heuristic: prioritize time_critical zones with highest severity
                zones = sorted(st["zones"], key=lambda z: (z["time_critical"], z["severity"]), reverse=True)
                
                deployments = []
                rem_a = res["ambulances"]
                rem_r = res["rescue_teams"]
                rem_f = res["food_packets"]

                for z in zones:
                    if z["contained"] or z["severity"] <= 0:
                        continue
                    need = z["resources_needed"]
                    # Deploy up to what's needed, capped by what's available
                    a = min(need["ambulances"], rem_a)
                    r_t = min(need["rescue_teams"], rem_r)
                    f = min(need["food_packets"], rem_f)
                    
                    if a > 0 or r_t > 0 or f > 0:
                        deployments.append({
                            "zone_id": z["zone_id"],
                            "ambulances": a,
                            "rescue_teams": r_t,
                            "food_packets": f,
                            "priority": 5 if z["time_critical"] else 3
                        })
                        rem_a -= a
                        rem_r -= r_t
                        rem_f -= f

                # Step
                step_resp = c.post("/step", json={"deployments": deployments}).json()
                step_num += 1
                reward = step_resp["reward"]
                score = step_resp["score"]
                done = step_resp["done"]

                zones_info = ", ".join(
                    f"{z['zone_id']}(sev={z['severity']:.2f} surv={z['survivors']})"
                    for z in step_resp["state"]["zones"]
                    if not z["contained"]
                )
                print(f"  Step {step_num}: reward={reward:+.4f} | score={score:.4f} | zones=[{zones_info}]")

                if done:
                    passed = score >= task["pass_threshold"]
                    status = "PASSED" if passed else "FAILED"
                    print(f"  >> Final score: {score:.4f} ({status}, threshold={task['pass_threshold']})")
                    break

            print()

if __name__ == "__main__":
    run_demo()
