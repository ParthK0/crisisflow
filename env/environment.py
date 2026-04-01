"""CrisisFlow OpenEnv-style sequential simulator."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Set, cast

import numpy as np
import yaml

from env.models import (
    Action,
    Deployment,
    DisasterState,
    DisasterType,
    DisasterZone,
    HealthResponse,
    ResetRequest,
    ResourceNeeds,
    ResourcePool,
    StateResponse,
    StepResult,
    TaskConfig,
)

_W1, _W2, _W3, _W4 = 0.50, 0.20, 0.15, 0.15


class CrisisFlowEnv:
    """Full CrisisFlow OpenEnv API: reset, step, state, list_tasks."""

    def __init__(self) -> None:
        self.current_state: DisasterState | None = None
        self.task_config: TaskConfig | None = None
        self.episode_log: List[Dict[str, Any]] = []
        self.rng: np.random.Generator | None = None
        self.tasks: Dict[str, TaskConfig] = {
            "task_easy": TaskConfig(
                id="task_easy",
                name="Localized incident",
                difficulty="easy",
                description="Single-zone flood scenario with tight time horizon.",
                max_steps=10,
                pass_threshold=0.75,
            ),
            "task_medium": TaskConfig(
                id="task_medium",
                name="Regional disruption",
                difficulty="medium",
                description="Three concurrent hazards with shared resources.",
                max_steps=15,
                pass_threshold=0.70,
            ),
            "task_hard": TaskConfig(
                id="task_hard",
                name="Systemic crisis",
                difficulty="hard",
                description="Five zones including a latent outbreak; cascade risk.",
                max_steps=20,
                pass_threshold=0.60,
            ),
        }
        self._initial_survivors: Dict[str, int] = {}
        self._initial_survivor_total: int = 0
        self._initial_pool_total: int = 0
        self._zone_spread: Dict[str, bool] = {}
        self._deployed_zones_this_step: Set[str] = set()
        self._zone_e_revealed: bool = False
        self._zone_e_reveal_step: int | None = None
        self._deployed_zone_e_within_grace: bool = False
        self._spread_this_step: bool = False
        self._last_reward: float | None = None
        self._done: bool = False
        self._last_grade_score: float = 0.0

    @property
    def last_reward(self) -> float | None:
        return self._last_reward

    @property
    def done(self) -> bool:
        return self._done

    @property
    def episode_score(self) -> float:
        """Latest aggregate grade from ``_grade_episode`` (same as ``StepResult.score``)."""

        return self._last_grade_score

    def reset(self, task_id: str = "task_easy", seed: int = 42) -> DisasterState:
        self.rng = np.random.default_rng(seed)
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task_id: {task_id!r}")
        self.task_config = self.tasks[task_id]
        self.episode_log = []
        self._zone_spread = {}
        self._deployed_zones_this_step = set()
        self._zone_e_revealed = False
        self._zone_e_reveal_step = None
        self._deployed_zone_e_within_grace = False
        self._spread_this_step = False
        self._last_reward = None
        self._done = False
        self._last_grade_score = 0.0

        self.current_state = self._generate_initial_state(task_id, seed)
        _ = yaml.safe_load("{}")
        _ = random.getrandbits(8)
        return self.current_state

    def _generate_initial_state(self, task_id: str, seed: int) -> DisasterState:
        self._initial_survivors = {}
        zones: List[DisasterZone] = []

        if task_id == "task_easy":
            zones = [
                DisasterZone(
                    zone_id="z1",
                    disaster_type="flood",
                    severity=0.7,
                    survivors=50,
                    casualties=0,
                    resources_needed=ResourceNeeds(ambulances=4, rescue_teams=3, food_packets=15),
                    time_critical=True,
                    accessibility=0.9,
                    contained=False,
                )
            ]
            pool = ResourcePool(ambulances=8, rescue_teams=6, food_packets=30)
            time_remaining = 10

        elif task_id == "task_medium":
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
            pool = ResourcePool(ambulances=8, rescue_teams=8, food_packets=40)
            time_remaining = 15

        elif task_id == "task_hard":
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
            pool = ResourcePool(ambulances=15, rescue_teams=12, food_packets=60)
            time_remaining = 20
        else:
            raise ValueError(f"Unknown task_id: {task_id!r}")

        for z in zones:
            self._zone_spread[z.zone_id] = False
            self._initial_survivors[z.zone_id] = z.survivors

        self._initial_survivor_total = sum(
            s for zid, s in self._initial_survivors.items() if zid != "E" or s > 0
        )
        self._initial_pool_total = pool.ambulances + pool.rescue_teams + pool.food_packets

        return DisasterState(
            zones=zones,
            resources=pool,
            time_remaining=time_remaining,
            step_count=0,
            cumulative_reward=0.0,
            task_id=task_id,
            seed=seed,
        )

    def step(self, action: Action) -> StepResult:
        if self.current_state is None:
            raise RuntimeError("Call reset() first")
        assert self.rng is not None and self.task_config is not None
        st = self.current_state
        zones_before = copy.deepcopy(st.zones)

        total_a = sum(d.ambulances for d in action.deployments)
        total_r = sum(d.rescue_teams for d in action.deployments)
        total_f = sum(d.food_packets for d in action.deployments)
        if total_a > st.resources.ambulances:
            raise ValueError(
                f"Ambulances deployed ({total_a}) exceed available ({st.resources.ambulances})"
            )
        if total_r > st.resources.rescue_teams:
            raise ValueError(
                f"Rescue teams deployed ({total_r}) exceed available ({st.resources.rescue_teams})"
            )
        if total_f > st.resources.food_packets:
            raise ValueError(
                f"Food packets deployed ({total_f}) exceed available ({st.resources.food_packets})"
            )

        self._deployed_zones_this_step = set()
        self._spread_this_step = False
        for dep in sorted(action.deployments, key=lambda x: x.priority, reverse=True):
            zone = next((z for z in st.zones if z.zone_id == dep.zone_id), None)
            if zone is None:
                continue
            self._deployed_zones_this_step.add(zone.zone_id)
            st.resources.ambulances -= dep.ambulances
            st.resources.rescue_teams -= dep.rescue_teams
            st.resources.food_packets -= dep.food_packets
            self._apply_deployment(zone, dep)

        self._simulate_tick()

        zones_after = copy.deepcopy(st.zones)
        reward = self._calculate_reward(action, zones_before, zones_after)
        reward = float(np.clip(reward, -0.2, 1.0))

        st.step_count += 1
        st.time_remaining -= 1
        st.cumulative_reward += reward

        if st.task_id == "task_hard" and st.step_count == 10:
            self._reveal_zone_e(st)

        if (
            st.task_id == "task_hard"
            and self._zone_e_revealed
            and self._zone_e_reveal_step is not None
            and self._zone_e_reveal_step < st.step_count <= self._zone_e_reveal_step + 2
            and "E" in self._deployed_zones_this_step
        ):
            self._deployed_zone_e_within_grace = True

        self._done = st.time_remaining <= 0 or all(z.contained for z in st.zones)
        self._last_reward = reward
        self._last_grade_score = self._grade_episode()

        log_entry: Dict[str, Any] = {
            "step": st.step_count,
            "action": action.model_dump(),
            "reward": reward,
            "state_snapshot": st.model_dump(),
        }
        if self._spread_this_step:
            log_entry["spread_event"] = True
        self.episode_log.append(log_entry)

        info: Dict[str, Any] = {
            "episode_log_len": len(self.episode_log),
            "spread_events": self._had_any_spread_so_far(),
        }

        return StepResult(
            state=copy.deepcopy(st),
            reward=round(reward, 4),
            done=self._done,
            score=self._last_grade_score,
            info=info,
        )

    def _had_any_spread_so_far(self) -> bool:
        if self._spread_this_step:
            return True
        return any(e.get("spread_event") for e in self.episode_log)

    def _reveal_zone_e(self, st: DisasterState) -> None:
        for z in st.zones:
            if z.zone_id != "E":
                continue
            z.disaster_type = cast(DisasterType, "disease")
            z.severity = 0.65
            z.survivors = 35
            z.casualties = 0
            z.resources_needed = ResourceNeeds(ambulances=2, rescue_teams=1, food_packets=20)
            z.time_critical = True
            z.contained = False
            self._zone_e_revealed = True
            self._zone_e_reveal_step = st.step_count
            self._initial_survivors["E"] = 35
            self._initial_survivor_total = sum(
                s for zid, s in self._initial_survivors.items() if zid != "E" or s > 0
            )
            break

    def _apply_deployment(self, zone: DisasterZone, deployment: Deployment) -> None:
        assert self.rng is not None
        need_a = max(1, zone.resources_needed.ambulances)
        eff = float(np.clip(deployment.ambulances / need_a, 0.0, 1.0))
        loss = int(self.rng.integers(0, 3) * (1.0 - eff))
        zone.survivors = max(0, zone.survivors - loss)
        zone.severity = max(0.0, zone.severity - eff * 0.15)
        zone.casualties += int(self.rng.integers(0, 2) * (1.0 - eff))
        if eff >= 0.8 and zone.severity < 0.2:
            zone.contained = True

    def _simulate_tick(self) -> None:
        assert self.current_state is not None and self.rng is not None
        st = self.current_state
        for zone in st.zones:
            if zone.contained:
                continue
            if st.task_id == "task_hard" and zone.zone_id == "E" and not self._zone_e_revealed:
                continue
            if zone.survivors == 0 and zone.severity <= 0.0:
                continue
            deployed_here = zone.zone_id in self._deployed_zones_this_step
            lo, hi = (0, 2) if deployed_here else (2, 8)
            loss = int(self.rng.integers(lo, hi))
            zone.survivors = max(0, zone.survivors - loss)
            zone.severity = min(1.0, zone.severity + 0.03)
            zone.casualties += loss
            if st.task_id == "task_hard" and zone.severity > 0.8 and not self._zone_spread.get(
                zone.zone_id, False
            ):
                self._zone_spread[zone.zone_id] = True
                self._spread_this_step = True

    def _calculate_reward(
        self,
        action: Action,
        zones_before: List[DisasterZone],
        zones_after: List[DisasterZone],
    ) -> float:
        init_total = max(1, self._initial_survivor_total)
        improv = sum(
            max(0, za.survivors - zb.survivors)
            for zb, za in zip(zones_before, zones_after, strict=True)
        )
        c1 = float(np.clip(improv / init_total, 0.0, 1.0))

        def zone_by_id(zs: List[DisasterZone], zid: str) -> DisasterZone | None:
            for z in zs:
                if z.zone_id == zid:
                    return z
            return None

        crit_ids = {z.zone_id for z in zones_before if z.time_critical}
        if not crit_ids:
            c2 = 0.0
        else:
            acc = 0.0
            for zid in crit_ids:
                deployed = zid in self._deployed_zones_this_step
                acc += 0.2 if deployed else -0.1
            c2 = acc / len(crit_ids)

        if not action.deployments:
            c3 = 0.0
        else:
            eff_scores: List[float] = []
            for dep in action.deployments:
                z = zone_by_id(zones_before, dep.zone_id)
                if z is None:
                    continue
                need_a = max(1, z.resources_needed.ambulances)
                need_r = max(1, z.resources_needed.rescue_teams)
                need_f = max(1, z.resources_needed.food_packets)
                ra = dep.ambulances / need_a
                rr = dep.rescue_teams / need_r
                rf = dep.food_packets / need_f
                pen = 0.0
                for ratio in (ra, rr, rf):
                    if ratio > 2.0:
                        pen += 0.1 * (ratio - 2.0)
                bonus = 0.0
                for ratio in (ra, rr, rf):
                    if 0.8 <= ratio <= 1.2:
                        bonus += 0.05
                eff_scores.append(
                    float(np.clip(0.5 * (ra + rr + rf) / 3.0 - pen + bonus, -1.0, 1.0))
                )
            c3 = float(np.mean(eff_scores)) if eff_scores else 0.0

        c4 = 0.0
        for zb, za in zip(zones_before, zones_after, strict=True):
            if zb.severity > 0.75 and za.severity < 0.7:
                c4 = 1.0
                break

        raw = _W1 * c1 + _W2 * c2 + _W3 * c3 + _W4 * c4
        return round(float(raw), 4)

    def _grade_episode(self) -> float:
        assert self.current_state is not None and self.task_config is not None
        st = self.current_state
        surv = sum(z.survivors for z in st.zones)
        denom = max(1, self._initial_survivor_total)
        survival_rate = surv / denom

        if st.task_id == "task_easy":
            sc = survival_rate * (
                1.0 + 0.2 * (1.0 - st.step_count / max(1, self.task_config.max_steps))
            )
            score = float(min(1.0, sc))
        elif st.task_id == "task_medium":
            wsum = 0.0
            ssum = 0.0
            for z in st.zones:
                init = max(1, self._initial_survivors.get(z.zone_id, z.survivors))
                w = 1.0 - z.severity
                wsum += w * z.survivors / init
                ssum += w
            weighted = wsum / max(1e-6, ssum)
            cur_pool = st.resources.ambulances + st.resources.rescue_teams + st.resources.food_packets
            efficiency = float(
                np.clip(
                    0.85 + 0.15 * (1.0 - cur_pool / max(1, self._initial_pool_total)),
                    0.5,
                    1.0,
                )
            )
            score = float(np.clip(weighted * efficiency * (0.98 + 0.04 * random.random()), 0.0, 1.0))
        else:
            base = float(np.clip(survival_rate * 0.85, 0.0, 1.0))
            cascade_bonus = 0.15 if not self._had_any_spread_so_far() else 0.0
            adapt_bonus = 0.10 if self._deployed_zone_e_within_grace else 0.0
            score = float(np.clip(base + cascade_bonus + adapt_bonus, 0.0, 1.0))

        return round(float(np.clip(score, 0.0, 1.0)), 4)

    def state(self) -> DisasterState:
        if self.current_state is None:
            raise RuntimeError("Call reset() first")
        return copy.deepcopy(self.current_state)

    def list_tasks(self) -> List[TaskConfig]:
        return [self.tasks[k] for k in ("task_easy", "task_medium", "task_hard")]


class CrisisEnvironment(CrisisFlowEnv):
    """Backwards-compatible name for HTTP and older imports."""

    def reset_from_request(self, req: ResetRequest) -> DisasterState:
        return self.reset(req.task_id, req.seed)

    def get_internal(self) -> CrisisFlowEnv:
        return self

    def disaster_state(self) -> DisasterState:
        return self.state()
