import mlflow
from mlflow.entities import ViewType
def main():
    experiment = mlflow.get_experiment_by_name("Breast_Cancer_Classification")
    if not experiment:
        print("Experiment not found. Please run training first.")
        return
    # Fetch all active runs, ordered by F1 Score descending
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["metrics.f1_score DESC"]
    )
    if len(runs) < 2:
        print("Need at least 2 runs in MLflow to perform a comparison.")
        return
    best_run = runs.iloc[0]
    runner_up = runs.iloc[1]
    print("=== MODEL COMPARISON REPORT ===")
    print(f"Top Model Run: {best_run['tags.mlflow.runName']}")
    print(f" - F1 Score : {best_run['metrics.f1_score']:.4f}")
    print(f" - Recall   : {best_run['metrics.recall']:.4f}")
    print(f" - Dataset  : {best_run['tags.dataset_fingerprint']}")
    print(f"\nRunner-up Run: {runner_up['tags.mlflow.runName']}")
    print(f" - F1 Score : {runner_up['metrics.f1_score']:.4f}")
    print(f" - Recall   : {runner_up['metrics.recall']:.4f}")
    diff_f1 = best_run['metrics.f1_score'] - runner_up['metrics.f1_score']
    print(f"\n[Verdict] Top model outperforms runner-up by {diff_f1:.4f} F1-score.")
if __name__ == "__main__":
    main()
