import os
import json
import mlflow
from mlflow.entities import ViewType
def main():
    experiment = mlflow.get_experiment_by_name("Breast_Cancer_Classification")
    if not experiment:
        print("Experiment not found.")
        return
    # Fetch active runs, ordered by F1
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.f1_score DESC"]
    )
    if len(runs) == 0:
        print("No runs found to evaluate.")
        return
    # Top run is considered the "candidate" for release
    candidate_run = runs.iloc[0]
    candidate_f1 = candidate_run['metrics.f1_score']
    candidate_recall = candidate_run['metrics.recall']
    run_name = candidate_run['tags.mlflow.runName']
    print(f"Evaluating Run: {run_name} for Release...")
    # Release Policy Thresholds
    MIN_F1_THRESHOLD = 0.95
    MIN_RECALL_THRESHOLD = 0.95
    is_approved = candidate_f1 >= MIN_F1_THRESHOLD and candidate_recall >= MIN_RECALL_THRESHOLD
    status = "APPROVED" if is_approved else "REJECTED"
    print(f"Status: {status}")
    print(f"F1 Score: {candidate_f1:.4f} (Required: {MIN_F1_THRESHOLD})")
    print(f"Recall:   {candidate_recall:.4f} (Required: {MIN_RECALL_THRESHOLD})")
    # Generate Release Report Evidence
    report = {
        "run_name": run_name,
        "run_id": candidate_run['run_id'],
        "dataset_fingerprint": candidate_run.get('tags.dataset_fingerprint', 'Unknown'),
        "release_status": status,
        "policy": {
            "min_f1_score": MIN_F1_THRESHOLD,
            "min_recall": MIN_RECALL_THRESHOLD
        },
        "metrics_evidence": {
            "f1_score": candidate_f1,
            "recall": candidate_recall
        }
    }
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/release_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nRelease report saved to {report_path}")
if __name__ == "__main__":
    main()
