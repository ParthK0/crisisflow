"""Pydantic v2 domain models for the CrisisFlow disaster-response environment."""

from typing import Any, Dict, List, Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

# --- 1 --- Literal union used as a typed disaster category (flood / earthquake / fire / disease).
DisasterType: TypeAlias = Literal["flood", "earthquake", "fire", "disease"]


# --- 2 ---
class ResourceNeeds(BaseModel):
    """Per-zone unfulfilled resource requirements (vehicles, teams, supplies)."""

    model_config = ConfigDict(use_enum_values=True)

    ambulances: int = Field(ge=0)
    rescue_teams: int = Field(ge=0)
    food_packets: int = Field(ge=0)


# --- 3 ---
class DisasterZone(BaseModel):
    """One affected geographic or logical area and its crisis metrics."""

    model_config = ConfigDict(use_enum_values=True)

    zone_id: str = Field(min_length=1)
    disaster_type: DisasterType
    severity: float = Field(ge=0.0, le=1.0)
    survivors: int = Field(ge=0)
    casualties: int = Field(ge=0)
    resources_needed: ResourceNeeds
    time_critical: bool = Field(default=False)
    accessibility: float = Field(default=1.0, ge=0.0, le=1.0)
    contained: bool = Field(default=False)


# --- 4 ---
class ResourcePool(BaseModel):
    """Counts of deployable units the agent controls globally."""

    model_config = ConfigDict(use_enum_values=True)

    ambulances: int = Field(ge=0)
    rescue_teams: int = Field(ge=0)
    food_packets: int = Field(ge=0)


# --- 5 ---
class DisasterState(BaseModel):
    """Full environment observation: all zones, pool, clock, and run bookkeeping."""

    model_config = ConfigDict(use_enum_values=True)

    zones: List[DisasterZone]
    resources: ResourcePool
    time_remaining: int
    step_count: int
    cumulative_reward: float = Field(default=0.0)
    task_id: str
    seed: int


# --- 6 ---
class Deployment(BaseModel):
    """A single dispatch order routing pool resources to one zone."""

    model_config = ConfigDict(use_enum_values=True)

    zone_id: str = Field(min_length=1)
    ambulances: int = Field(default=0, ge=0)
    rescue_teams: int = Field(default=0, ge=0)
    food_packets: int = Field(default=0, ge=0)
    priority: int = Field(default=3, ge=1, le=5)


# --- 7 ---
class Action(BaseModel):
    """Agent decision: zero or more deployments, applied in priority order."""

    model_config = ConfigDict(use_enum_values=True)

    deployments: List[Deployment] = Field(default_factory=list)


# --- 8 ---
class StepResult(BaseModel):
    """Outcome of one environment transition after applying an Action."""

    model_config = ConfigDict(use_enum_values=True)

    state: DisasterState
    reward: float
    done: bool
    score: float
    info: Dict[str, Any]


# --- 9 ---
class TaskConfig(BaseModel):
    """Registered benchmark scenario (length, difficulty, success bar)."""

    model_config = ConfigDict(use_enum_values=True)

    id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    max_steps: int
    pass_threshold: float


# --- 10 ---
class ResetRequest(BaseModel):
    """Client request to start or restart an episode."""

    model_config = ConfigDict(use_enum_values=True)

    task_id: str = Field(default="task_easy")
    seed: int = Field(default=42)


class StateResponse(BaseModel):
    """API envelope for /reset and /state with episode flags."""

    model_config = ConfigDict(use_enum_values=True)

    state: DisasterState
    last_reward: Optional[float] = None
    score: float = 0.0
    done: bool = False


class HealthResponse(BaseModel):
    """Minimal liveness payload for container health checks."""

    model_config = ConfigDict(use_enum_values=True)

    status: Literal["ok"] = "ok"
    version: str = "1.0.0"
