"""Standalone grader and initial scenario for task_easy (single-zone triage)."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from env.models import DisasterState, DisasterZone, ResourceNeeds, ResourcePool, TaskConfig

TASK_CONFIG = TaskConfig(
    id="task_easy",
    name="Single-Zone Triage",
    difficulty="easy",
    description="One flood zone. Allocate resources to reach 80% survival in 10 steps.",
    max_steps=10,
    pass_threshold=0.75,
)


def get_config() -> TaskConfig:
    return TASK_CONFIG


def initial_state(seed: int) -> DisasterState:
    """Deterministic single-zone flood state aligned with ``grade()`` keys."""
    _ = np.random.default_rng(seed)  # reserved for future stochastic init
    zone = DisasterZone(
        zone_id="zone_a",
        disaster_type="flood",
        severity=0.7,
        survivors=50,
        casualties=0,
        resources_needed=ResourceNeeds(ambulances=4, rescue_teams=3, food_packets=15),
        time_critical=True,
        accessibility=0.9,
        contained=False,
    )
    return DisasterState(
        zones=[zone],
        resources=ResourcePool(ambulances=8, rescue_teams=6, food_packets=30),
        time_remaining=10,
        step_count=0,
        cumulative_reward=0.0,
        task_id="task_easy",
        seed=seed,
    )


def grade(episode_log: List[Any], final_state: DisasterState, initial_survivors: dict) -> float:
    if not final_state.zones:
        return 0.0
    init_a = max(1, int(initial_survivors.get("zone_a", 1)))
    survival_rate = final_state.zones[0].survivors / init_a
    steps_used = final_state.step_count
    time_bonus = 1.0 + 0.2 * (1.0 - steps_used / 10.0)
    score = min(1.0, survival_rate * time_bonus)
    return round(float(score), 4)
