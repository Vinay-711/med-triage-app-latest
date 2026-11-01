# MedTriage AI — Triage Workflow Demonstration

A React/TypeScript interface and Python API exploring radiology and dermatology triage workflows. The project includes simulation paths; generated confidence values and visual overlays must not be interpreted as clinically validated findings.

[Browser demonstration](https://vinay-711.github.io/med-triage-app-latest/)

## Run the frontend

```sh
cd frontend
npm ci
npm run dev
```

Use the address printed by Vite. The deployed static page is a frontend demonstration and cannot host the Python inference service.

## Run the backend separately

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The dependency list includes ML frameworks; confirm compatible Python/platform wheels before installation. Inspect [backend/main.py](backend/main.py) for model-loading, provider configuration and simulation behavior. Training scripts are separate experiments and do not establish model validity.

## Source map

| File | Role |
| --- | --- |
| [frontend/src/TriageApp.tsx](frontend/src/TriageApp.tsx) | Triage interface and client-side interactions |
| [backend/main.py](backend/main.py) | FastAPI application and analysis behavior |
| [backend/train.py](backend/train.py) | Training experiment |
| [backend/kaggle_training_script.py](backend/kaggle_training_script.py) | Dataset/training experiment |

## Verification and limitations

Run `npm run build` from `frontend` to check TypeScript and the production bundle. No clinical accuracy or automated test result is claimed here. Review simulation labels, invalid uploads, empty queues and API-offline behavior manually. Use synthetic examples, keep patient images out of Git, and require qualified human review for any health interpretation.
