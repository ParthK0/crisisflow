"""Standalone grader and initial scenario for task_medium (multi-zone allocation)."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from env.models import DisasterState, DisasterZone, ResourceNeeds, ResourcePool, TaskConfig

TASK_CONFIG = TaskConfig(
    id="task_medium",
    name="Multi-Zone Resource Allocation",
    difficulty="medium",
    description="Three simultaneous zones. Prioritise correctly across flood, earthquake, fire.",
    max_steps=15,
    pass_threshold=0.70,
)

# Initial pool from ``initial_state`` (sum of all deployable units at episode start).
INITIAL_RESOURCE_TOTAL = 8 + 8 + 40


def get_config() -> TaskConfig:
    return TASK_CONFIG


def initial_state(seed: int) -> DisasterState:
    _ = np.random.default_rng(seed)
    zones = [
        DisasterZone(
            zone_id="A",
            disaster_type="earthquake",
            severity=0.85,
            survivors=80,
            casualties=5,
            resources_needed=ResourceNeeds(ambulances=6, rescue_teams=5, food_packets=20),
            time_critical=True,
            accessibility=1.0,
            contained=False,
        ),
        DisasterZone(
            zone_id="B",
            disaster_type="fire",
            severity=0.65,
            survivors=40,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=2, rescue_teams=4, food_packets=10),
            time_critical=False,
            accessibility=1.0,
            contained=False,
        ),
        DisasterZone(
            zone_id="C",
            disaster_type="flood",
            severity=0.5,
            survivors=60,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=3, rescue_teams=2, food_packets=25),
            time_critical=False,
            accessibility=1.0,
            contained=False,
        ),
    ]
    return DisasterState(
        zones=zones,
        resources=ResourcePool(ambulances=8, rescue_teams=8, food_packets=40),
        time_remaining=15,
        step_count=0,
        cumulative_reward=0.0,
        task_id="task_medium",
        seed=seed,
    )


def _total_wasted_from_log(episode_log: List[Any]) -> float:
    """Sum explicit ``over_deployments`` fields; optional per-step records from environment."""
    total = 0.0
    for entry in episode_log:
        if not isinstance(entry, dict):
            continue
        if "over_deployments" in entry:
            total += float(entry["over_deployments"])
    return total


def grade(episode_log: List[Any], final_state: DisasterState, initial_survivors: dict) -> float:
    if not final_state.zones:
        return 0.0
    sev_sum = sum(z.severity for z in final_state.zones)
    if sev_sum <= 0:
        return 0.0
    weighted_num = 0.0
    for z in final_state.zones:
        init = max(1, int(initial_survivors.get(z.zone_id, 1)))
        zone_score = z.survivors / init
        weighted_num += z.severity * zone_score
    weighted_score = weighted_num / sev_sum

    total_wasted = _total_wasted_from_log(episode_log)
    denom = max(1, INITIAL_RESOURCE_TOTAL)
    efficiency_penalty = min(0.3, (total_wasted / denom) * 0.3)
    score = min(1.0, weighted_score - efficiency_penalty)
    return round(float(score), 4)
