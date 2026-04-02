---
title: CrisisFlow
emoji: 🔥
colorFrom: red
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# CrisisFlow â€” AI Disaster Response Environment

CrisisFlow is an OpenEnv environment where an AI agent acts as an emergency operations commander allocating limited resources across disaster zones to maximise survivor outcomes. The agent must balance competing priorities, manage dwindling supplies, and respond to dynamic, multi-hazard scenarios ranging from urban floods to systemic health crises.

## Environment description
The simulation operates on a discrete-time basis. Disasters spawn across multiple geographical zones, each with unique characteristics such as severity, survivor count, and resource requirements. The agent has a central pool of resources (ambulances, rescue teams, and food packets). In each step, the agent dispatches these resources to specific zones. The world then "ticks" forward: resources reduce disaster severity and save lives, while unaddressed disasters worsen, leading to casualties and potential hazard spread. This environment mimics the high-stakes decision-making required by agencies like NDRF or FEMA.

## Observation space
| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `zones` | `array` | List of active disaster zones | - |
| `zone.zone_id` | `string` | Unique identifier for the zone | - |
| `zone.disaster_type` | `string` | Type of disaster (flood, fire, earthquake, disease) | - |
| `zone.severity` | `float` | Intensity of the disaster | 0.0 - 1.0 |
| `zone.survivors` | `int` | Number of people needing rescue | 0 - 500+ |
| `zone.casualties` | `int` | Cumulative deaths in this zone | 0+ |
| `zone.resources_needed` | `object` | Required ambulances, rescue_teams, food_packets | - |
| `zone.time_critical` | `boolean` | Whether the zone requires immediate attention | True/False |
| `zone.accessibility` | `float` | Multiplier for resource effectiveness | 0.0 - 1.0 |
| `zone.contained` | `boolean` | Whether the zone is contained | True/False |
| `resources.ambulances` | `int` | Remaining ambulances in the pool | 0+ |
| `resources.rescue_teams` | `int` | Remaining rescue teams in the pool | 0+ |
| `resources.food_packets` | `int` | Remaining food packets in the pool | 0+ |
| `time_remaining` | `int` | Steps left in the episode | 0 - 20 |
| `step_count` | `int` | Number of steps taken so far | 0 - 20 |
| `cumulative_reward` | `float` | Total reward earned in the episode | - |
| `task_id` | `string` | Active task key (e.g. task_easy) | - |
| `seed` | `int` | PRNG seed for this episode | - |

## Action space
| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| `deployments` | `array` | List of resource allocations to zones | - |
| `deployment.zone_id` | `string` | Target zone for the resources | Must exist in state |
| `deployment.ambulances` | `int` | Number of ambulances to send | â‰¥ 0, sum â‰¤ pool |
| `deployment.rescue_teams` | `int` | Number of rescue teams to send | â‰¥ 0, sum â‰¤ pool |
| `deployment.food_packets` | `int` | Number of food packets to send | â‰¥ 0, sum â‰¤ pool |
| `deployment.priority` | `int` | Processing order for the deployment | 1 - 5 |

## Tasks
| Task ID | Difficulty | Max Steps | Pass Threshold | Description |
|---------|------------|-----------|----------------|-------------|
| `task_easy` | easy | 10 | 0.75 | Single-zone flood scenario with tight time horizon. |
| `task_medium` | medium | 15 | 0.70 | Three concurrent hazards with shared resources. |
| `task_hard` | hard | 20 | 0.60 | Five zones including a latent outbreak; cascade risk. |

## Reward function
| Component | Weight | What it measures | Good behaviour it encourages |
|-----------|--------|------------------|------------------------------|
| Survivors Saved | 0.50 | Reduction in harm / effective aid to populations at risk | Saving lives quickly and effectively |
| Response Time | 0.20 | Penalize delay; prioritize time-critical zones | Prioritizing zones where time is of the essence |
| Resource Efficiency | 0.15 | Favor useful deployments versus waste or idle pool | Optimal allocation without waste or shortage |
| Cascade Prevention | 0.15 | Limit severity growth and cross-zone escalation | Focusing on the most dangerous hazard zones |

## Setup instructions
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd crisisflow
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the environment server:
   ```bash
   uvicorn server.app:app --port 7860
   ```
4. In another terminal, run the baseline agent:
   ```bash
   export API_BASE_URL=http://localhost:7860
   export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
   export HF_TOKEN=your_token_here
   python inference.py
   ```

## Environment variables
| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | No | URL of the CrisisFlow server (default: http://localhost:7860) |
| `MODEL_NAME` | No | LLM model identifier (default: meta-llama/Llama-3.1-8B-Instruct) |
| `HF_TOKEN` | Yes | Hugging Face API token for LLM access |
| `LLM_BASE_URL` | No | Base URL for the LLM API (default: Hugging Face Inference API) |

## API endpoints
| Method | Endpoint | Description | Request body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/reset` | Reset the environment | `{"task_id": str, "seed": int}` | `StateResponse` |
| `GET` | `/state` | Get current state | - | `StateResponse` |
| `POST` | `/step` | Execute an action | `Action` object | `StepResult` |
| `GET` | `/tasks` | List available tasks | - | `List[TaskConfig]` |
| `GET` | `/health` | Check server status | - | `{"status": "ok"}` |

## Reproducible baseline scores
Expected output when running `python inference.py` with `seed=42`:
```text
Task easy   score: 0.82
Task medium score: 0.74
Task hard   score: 0.65
Average score: 0.74
```

## Baseline agent

With the server running:

```bash
python inference.py
```

## Validate

```bash
python validation/validate.py
```

## Docker

```bash
docker build -t crisisflow .
docker run -p 7860:7860 crisisflow
```

## Hugging Face Space
The environment and baseline agent can be viewed/deployed at: [https://huggingface.co/spaces/ParthK0/crisisflow](https://huggingface.co/spaces/ParthK0/crisisflow)


