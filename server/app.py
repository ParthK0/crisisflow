"""FastAPI service for the CrisisFlow OpenEnv environment."""

from __future__ import annotations

import logging
from typing import List

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from env.environment import CrisisFlowEnv
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

app = FastAPI(
    title="CrisisFlow",
    version="1.0.0",
    description="AI Disaster Response Environment",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = CrisisFlowEnv()
log = logging.getLogger("uvicorn.error")


@app.on_event("startup")
async def _startup() -> None:
    print("CrisisFlow environment ready", flush=True)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        return HealthResponse(status="ok", version="1.0.0")
    except Exception as exc:
        log.exception("health failed")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while handling the request.",
        ) from exc


@app.get("/tasks", response_model=List[TaskConfig])
def tasks() -> List[TaskConfig]:
    try:
        return env.list_tasks()
    except Exception as exc:
        log.exception("tasks failed")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while listing tasks.",
        ) from exc


@app.post("/reset", response_model=StateResponse)
def reset(
    req: ResetRequest = Body(),
) -> StateResponse:
    try:
        s = env.reset(task_id=req.task_id, seed=req.seed)
        return StateResponse(state=s, done=False, score=0.0)
    except ValueError:
        raise HTTPException(status_code=400, detail="Unknown task_id") from None
    except Exception as exc:
        log.exception("reset failed")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during reset.",
        ) from exc


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return env.step(action)
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Call /reset first") from None
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc) or "Invalid action",
        ) from exc
    except Exception as exc:
        log.exception("step failed")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during step.",
        ) from exc


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    try:
        s = env.state()
        return StateResponse(
            state=s,
            done=env.done,
            score=env.episode_score,
            last_reward=env.last_reward,
        )
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Call /reset first") from None
    except Exception as exc:
        log.exception("state failed")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while reading state.",
        ) from exc
