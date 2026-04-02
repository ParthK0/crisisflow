import os
import math
from env.environment import CrisisFlowEnv, TaskConfig
from env.models import Action, Deployment

def smart_agent_action(state, step_num, prev_zones):
    zones = state.zones
    res = state.resources
    time_remaining = state.time_remaining
    task_id = state.task_id

    # 4. ADAPT RULE
    new_zones = set()
    for z in zones:
        zid = z.zone_id
        if prev_zones.get(zid, 0) == 0 and z.survivors > 0:
            new_zones.add(zid)

    active_zones = [z for z in zones if z.survivors > 0 and not (z.contained and z.severity < 0.2)]
    if not active_zones:
        return Action(deployments=[])

    # 2. TRIAGE RULE
    scored = []
    for z in active_zones:
        sev = z.severity
        tc = z.time_critical
        surv = z.survivors
        triage_score = sev * (2.0 if tc else 1.0) * (surv / 100.0)
        if z.zone_id in new_zones:
            triage_score += 1000.0
        scored.append((triage_score, z))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    max_focus = 3 if task_id == "task_hard" else 2
    focus_zones = [z for _, z in scored[:max_focus]]

    # 3. RESERVE RULE
    if time_remaining <= 3:
        reserve_a, reserve_r, reserve_f = 0, 0, 0
    else:
        reserve_a, reserve_r, reserve_f = 2, 2, 5

    avail_a = max(0, res.ambulances - reserve_a)
    avail_r = max(0, res.rescue_teams - reserve_r)
    avail_f = max(0, res.food_packets - reserve_f)

    # 1. PACING RULE
    max_a = avail_a * 0.4
    max_r = avail_r * 0.4
    max_f = avail_f * 0.4

    deps = []
    deployed_ids = set()

    for z in focus_zones:
        need_a = z.resources_needed.ambulances
        need_r = z.resources_needed.rescue_teams
        need_f = z.resources_needed.food_packets
        
        dep_a = int(math.ceil(min(need_a, max_a)))
        dep_r = int(math.ceil(min(need_r, max_r)))
        dep_f = int(math.ceil(min(need_f, max_f)))

        if z.time_critical:
            if avail_a > 0 and dep_a == 0: dep_a = 1
            if avail_r > 0 and dep_r == 0: dep_r = 1
            if avail_f > 0 and dep_f == 0: dep_f = 1

        dep_a = min(dep_a, avail_a)
        dep_r = min(dep_r, avail_r)
        dep_f = min(dep_f, avail_f)

        deps.append(Deployment(
            zone_id=z.zone_id, ambulances=dep_a, rescue_teams=dep_r, food_packets=dep_f,
            priority=5 if z.time_critical else 3
        ))
        deployed_ids.add(z.zone_id)
        avail_a -= dep_a
        avail_r -= dep_r
        avail_f -= dep_f
        max_a = max(0, max_a - dep_a)
        max_r = max(0, max_r - dep_r)
        max_f = max(0, max_f - dep_f)

    # FILLER FIX (Missing from literal rules, but obviously necessary to prevent instant death)
    for z in active_zones:
        if z.zone_id not in deployed_ids:
            deps.append(Deployment(
                zone_id=z.zone_id, ambulances=0, rescue_teams=0, food_packets=0, priority=1
            ))

    return Action(deployments=deps)

def run():
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        env = CrisisFlowEnv()
        obs = env.reset(task_id=task_id, seed=42)
        prev_zones = {}
        state = obs
        while not state.done:
            action = smart_agent_action(state, state.step_count+1, prev_zones)
            prev_zones = {z.zone_id: z.survivors for z in state.zones}
            obs = env.step(action)
            state = obs
        print(f"{task_id}: {state.score:.4f}")

if __name__ == "__main__":
    run()
