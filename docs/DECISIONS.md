# Architecture & Technical Decisions
## 1. Dataset Selection
- **Decision:** Use Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn.
- **Rationale:** Small, public domain, tabular dataset ideal for demonstrating ML lifecycle, metrics (Precision/Recall), and drift without requiring GPUs or large storage.
## 2. Tracking Tool
- **Decision:** MLflow (Local).
- **Rationale:** Standard for experiment tracking; runs entirely offline initially to meet MVP requirements.
