import os
import json
import httpx
import time
from openai import OpenAI

# Read env vars
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api-inference.huggingface.co/v1")

client = OpenAI(base_url=LLM_BASE_URL, api_key=HF_TOKEN)

def format_state_for_llm(state: dict) -> str:
    """Format the state as readable text showing: step count, time remaining, available resources, each zone with its disaster type / severity / survivors / resources needed / time_critical status."""
    lines = []
    lines.append(f"Step Count: {state.get('step_count', 0)}")
    lines.append(f"Time Remaining: {state.get('time_remaining', 0)}")
    
    res = state.get('resources', {})
    lines.append(f"Available Resources: Ambulances: {res.get('ambulances', 0)}, Rescue Teams: {res.get('rescue_teams', 0)}, Food Packets: {res.get('food_packets', 0)}")
    
    lines.append("\nZones:")
    for zone in state.get('zones', []):
        zn_id = zone.get('zone_id')
        dtype = zone.get('disaster_type')
        sev = zone.get('severity')
        sur = zone.get('survivors')
        needs = zone.get('resources_needed', {})
        tc = "YES" if zone.get('time_critical') else "NO"
        
        lines.append(f"- Zone {zn_id}: {dtype} (Severity: {sev}) | Survivors: {sur} | Needs: {needs} | Time Critical: {tc}")
    
    return "\n".join(lines)

def parse_llm_action(response_text: str, state: dict) -> dict:
    """Try to parse JSON from the LLM response. Look for JSON between curly braces. If parsing fails, create a fallback action that deploys half of available resources to the most severe zone. Always validate that deployments don't exceed available resources — clip if needed."""
    try:
        # Look for JSON between curly braces
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end != 0:
            action_data = json.loads(response_text[start:end])
        else:
            raise ValueError("No JSON found")
    except (json.JSONDecodeError, ValueError):
        # Fallback action: deploy half of available resources to the most severe zone
        res = state.get('resources', {})
        zones = state.get('zones', [])
        if not zones:
            return {"deployments": []}
            
        target_zone = max(zones, key=lambda z: z.get('severity', 0))
        action_data = {
            "deployments": [{
                "zone_id": target_zone['zone_id'],
                "ambulances": int(res.get('ambulances', 0) * 0.5),
                "rescue_teams": int(res.get('rescue_teams', 0) * 0.5),
                "food_packets": int(res.get('food_packets', 0) * 0.5),
                "priority": 5
            }]
        }
    
    # Validate and clip deployments
    res_pool = state.get('resources', {})
    total_a = 0
    total_r = 0
    total_f = 0
    
    valid_deployments = []
    for d in action_data.get("deployments", []):
        a = max(0, int(d.get("ambulances", 0)))
        r = max(0, int(d.get("rescue_teams", 0)))
        f = max(0, int(d.get("food_packets", 0)))
        
        # Simple clipping to avoid over-deployment across multiple zones
        if total_a + a > res_pool.get("ambulances", 0):
            a = max(0, res_pool.get("ambulances", 0) - total_a)
        if total_r + r > res_pool.get("rescue_teams", 0):
            r = max(0, res_pool.get("rescue_teams", 0) - total_r)
        if total_f + f > res_pool.get("food_packets", 0):
            f = max(0, res_pool.get("food_packets", 0) - total_f)
            
        total_a += a
        total_r += r
        total_f += f
        
        valid_deployments.append({
            "zone_id": d.get("zone_id"),
            "ambulances": a,
            "rescue_teams": r,
            "food_packets": f,
            "priority": d.get("priority", 3)
        })
        
    return {"deployments": valid_deployments}

def call_llm(prompt: str, system: str) -> str:
    """Call client.chat.completions.create with model=MODEL_NAME, messages=[system, user], max_tokens=500, temperature=0.1. Return the content string. Wrap in try/except, return empty string on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"LLM call failed: {e}")
        return ""

def run_task(task_id: str, base_url: str) -> float:
    """Run a single task. Handle connection errors gracefully with retries (3 attempts, 2 second wait)."""
    url = base_url.rstrip("/")
    with httpx.Client(timeout=30.0) as http_client:
        # 1. POST {base_url}/reset with {"task_id": task_id, "seed": 42}
        for attempt in range(3):
            try:
                resp = http_client.post(f"{url}/reset", json={"task_id": task_id, "seed": 42})
                resp.raise_for_status()
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                if attempt == 2: 
                    print(f"Failed to reset task {task_id}: {e}")
                    return 0.0
                time.sleep(2)
        
        # 2. Print "Running {task_id}..."
        print(f"Running {task_id}...")
        
        # 3. Loop
        while True:
            # a. GET {base_url}/state
            state_data = None
            for attempt in range(3):
                try:
                    resp = http_client.get(f"{url}/state")
                    resp.raise_for_status()
                    state_data = resp.json()
                    break
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == 2: 
                        print(f"Failed to get state for {task_id}: {e}")
                        return 0.0
                    time.sleep(2)
            
            # i. If response["done"] is True: print final score, return response["score"]
            if state_data.get("done"):
                print(f"Final score for {task_id}: {state_data.get('score', 0.0):.2f}")
                return state_data.get("score", 0.0)
            
            state = state_data.get("state", {})
            
            # b. Format state as text
            state_text = format_state_for_llm(state)
            
            # c. Build system prompt
            system_prompt = (
                "You are a disaster response commander. Given the current disaster state and your available resources, "
                "decide how to deploy resources. Output ONLY a JSON object with key deployments, which is a list of "
                "objects each with: zone_id (string), ambulances (int), rescue_teams (int), food_packets (int), priority (int 1-5). "
                "Deploy only what is available. Prioritise time_critical zones and highest severity."
            )
            
            # d. Build user prompt using format_state_for_llm (handled in state_text)
            
            # e. Call call_llm
            response_text = call_llm(state_text, system_prompt)
            
            # f. Parse action using parse_llm_action
            action = parse_llm_action(response_text, state)
            
            # g. POST {base_url}/step with the action
            for attempt in range(3):
                try:
                    resp = http_client.post(f"{url}/step", json=action)
                    resp.raise_for_status()
                    step_resp = resp.json()
                    
                    # h. Print step reward from response
                    reward = step_resp.get('reward', 0.0)
                    print(f"Step Reward: {reward:.2f}")
                    
                    # i. If response["done"] is True: print final score, return response["score"]
                    if step_resp.get("done"):
                        score = step_resp.get("score", 0.0)
                        print(f"Final score for {task_id}: {score:.2f}")
                        return score
                    break
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt == 2: 
                        print(f"Failed to step for {task_id}: {e}")
                        return 0.0
                    time.sleep(2)
            
            # j. time.sleep(0.5) to avoid rate limits
            time.sleep(0.5)
            
    return 0.0

def main():
    scores = {}
    for task_id in ["task_easy", "task_medium", "task_hard"]:
        score = run_task(task_id, API_BASE_URL)
        scores[task_id] = score
        
    print("\n" + "="*30)
    print(f"Task easy   score: {scores.get('task_easy', 0.0):.2f}")
    print(f"Task medium score: {scores.get('task_medium', 0.0):.2f}")
    print(f"Task hard   score: {scores.get('task_hard', 0.0):.2f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"Average score: {avg:.2f}")
    print("="*30)

if __name__ == "__main__":
    main()
