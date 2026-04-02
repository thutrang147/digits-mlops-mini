# Digits MLOps Mini

A mini MLOps project for learning: train a digit classification model on the scikit-learn Digits dataset and serve predictions with FastAPI.

## What it does

- Trains a Logistic Regression model.
- Saves the model to artifacts/model.pkl.
- Exposes a simple API:
  - GET /health
  - POST /predict (expects 64 features)
- Includes tests and basic CI/CD workflows.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs at: http://127.0.0.1:8000/docs

## Run tests

```bash
pytest -q
```

## Docker

```bash
docker build -t digits-mlops-mini:latest .
docker run --rm -p 8000:8000 digits-mlops-mini:latest
```

## CI/CD

- CI: installs dependencies, trains model, runs tests.
- CD: builds and pushes image to GHCR on push to main.

## Note

If artifacts/model.pkl is missing, run:

```bash
python src/train.py
```
