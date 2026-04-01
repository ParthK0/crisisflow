"""Standalone grader and initial scenario for task_hard (cascading multi-disaster)."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from env.models import DisasterState, DisasterZone, ResourceNeeds, ResourcePool, TaskConfig

TASK_CONFIG = TaskConfig(
    id="task_hard",
    name="Cascading Multi-Disaster Coordination",
    difficulty="hard",
    description="Five zones with cascading failures and mid-episode surprises.",
    max_steps=20,
    pass_threshold=0.60,
)

ZONE_E_APPEARED_AT_STEP = 10


def get_config() -> TaskConfig:
    return TASK_CONFIG


def initial_state(seed: int) -> DisasterState:
    _ = np.random.default_rng(seed)
    zones = [
        DisasterZone(
            zone_id="A",
            disaster_type="earthquake",
            severity=0.9,
            survivors=100,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=8, rescue_teams=6, food_packets=30),
            time_critical=True,
            accessibility=1.0,
            contained=False,
        ),
        DisasterZone(
            zone_id="B",
            disaster_type="flood",
            severity=0.6,
            survivors=70,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=4, rescue_teams=3, food_packets=20),
            time_critical=False,
            accessibility=1.0,
            contained=False,
        ),
        DisasterZone(
            zone_id="C",
            disaster_type="fire",
            severity=0.75,
            survivors=55,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=3, rescue_teams=5, food_packets=15),
            time_critical=True,
            accessibility=1.0,
            contained=False,
        ),
        DisasterZone(
            zone_id="D",
            disaster_type="flood",
            severity=0.4,
            survivors=40,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=2, rescue_teams=2, food_packets=10),
            time_critical=False,
            accessibility=1.0,
            contained=False,
        ),
        DisasterZone(
            zone_id="E",
            disaster_type="flood",
            severity=0.0,
            survivors=0,
            casualties=0,
            resources_needed=ResourceNeeds(ambulances=0, rescue_teams=0, food_packets=0),
            time_critical=False,
            accessibility=1.0,
            contained=False,
        ),
    ]
    return DisasterState(
        zones=zones,
        resources=ResourcePool(ambulances=15, rescue_teams=12, food_packets=60),
        time_remaining=20,
        step_count=0,
        cumulative_reward=0.0,
        task_id="task_hard",
        seed=seed,
    )


def _first_zone_e_deploy_step(episode_log: List[Any]) -> Optional[int]:
    for entry in episode_log:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action") or {}
        for dep in action.get("deployments") or []:
            if isinstance(dep, dict) and dep.get("zone_id") == "E":
                step = entry.get("step")
                if step is not None:
                    return int(step)
    return None


def grade(episode_log: List[Any], final_state: DisasterState, initial_survivors: dict) -> float:
    sev_sum = 0.0
    weighted_num = 0.0
    for z in final_state.zones:
        init = int(initial_survivors.get(z.zone_id, 0))
        if init <= 0:
            continue
        sev_sum += z.severity
        weighted_num += z.severity * (z.survivors / max(1, init))
    base_survival = weighted_num / sev_sum if sev_sum > 0 else 0.0

    cascade_bonus = 0.15 if not any(
        isinstance(e, dict) and e.get("spread_event") for e in episode_log
    ) else 0.0

    first_e = _first_zone_e_deploy_step(episode_log)
    adapt_bonus = 0.0
    if first_e is not None and (first_e - ZONE_E_APPEARED_AT_STEP) <= 2:
        adapt_bonus = 0.10

    score = min(1.0, base_survival + cascade_bonus + adapt_bonus)
    return round(float(score), 4)
