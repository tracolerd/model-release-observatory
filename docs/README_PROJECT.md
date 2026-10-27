# Model Release Observatory

A local-first reproducible ML model lifecycle demonstrator. This project establishes a pipeline from baseline training to MLflow tracking, automated release decision making, distribution drift checking, and inference latency benchmarking.

## Core Features
1. **Reproducible Baseline:** Stratified split, random seeds, and strict no-leakage tests.
2. **MLflow Tracking:** Parameter, metric, and artifact (model) logging with dataset fingerprinting.
3. **Explicit Release Workflow:** Deterministic JSON audit trail based on F1 and Recall thresholds.
4. **Drift Observatory:** Univariate distribution drift detection using the Kolmogorov-Smirnov test, with documented limitations and synthetic shift tests.
5. **Inference & Benchmarking:** Dynamic loading of the best approved model via FastAPI, coupled with a P50/P90/P99 latency benchmarking harness.

## Tech Stack
- **Python 3.10+
- **scikit-learn** & **pandas** (Modeling and Data)
- **MLflow** (Experiment Tracking & Artifact Lineage)
- **FastAPI** (Inference Endpoint)
- **pytest** (Unit Testing)

## Setup & Run Instructions

### 1. Environment Setup
```bash
python -m venv venv
# Activate on Windows
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m pytest tests/
```

### 3. Run Experiments (Train & Track)
```bash
python src/train.py --run_name "baseline_v1" --n_estimators 100 --max_depth 5
python src/train.py --run_name "candidate_v1" --n_estimators 200 --max_depth 10
```

### 4. Evaluate Runs & Generate Release Report
```bash
python src/compare_runs.py
python src/release.py
```

### 5. Check for Data Drift
```bash
python src/drift_check.py
```

### 6. Start Inference Server & Run Benchmark
Start the server in one terminal:
```bash
uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Run the benchmark in another terminal:
```bash
python src/latency_benchmark.py
```

## Security & Provenance Notes
- No secrets, private credentials, or large artifacts are committed to the repository.
- The `mlruns/` directory (SQLite DB and artifacts) is ignored via `.gitignore` to prevent accidental pushing of local environment states.
- The `scikit-learn` model artifact serialization utilizes `cloudpickle` as a safe fallback for local development.
