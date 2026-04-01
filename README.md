# CrisisFlow

Sequential crisis-management environment with an HTTP API compatible with the OpenEnv-style manifest in `openenv.yaml`.

## Layout

- `openenv.yaml` — specification (endpoints, spaces, tasks, Docker hints)
- `inference.py` — baseline loop against the local API
- `env/` — Pydantic models, `CrisisEnvironment`, and task modules
- `server/app.py` — FastAPI server (`/reset`, `/step`, `/state`, `/tasks`, `/health`)
- `validation/validate.py` — quick pre-submission checks

## Run locally

```bash
pip install -r requirements.txt
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
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
