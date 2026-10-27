import argparse
import hashlib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataset import get_train_test_split
def get_data_fingerprint(X):
    """Generate a simple MD5 hash fingerprint of the feature matrix."""
    return hashlib.md5(X.to_numpy().tobytes()).hexdigest()
def main():
    parser = argparse.ArgumentParser(description="Train and Track RF Model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the MLflow run")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    print(f"Starting run: {args.run_name}")
    X_train, X_test, y_train, y_test = get_train_test_split()
    fingerprint = get_data_fingerprint(X_train)
    mlflow.set_experiment("Breast_Cancer_Classification")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("dataset_fingerprint", fingerprint)
        mlflow.set_tag("dataset_shape", f"{X_train.shape}")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", 42)
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba))
        }
        mlflow.log_metrics(metrics)
        # BUGFIX: Specify serialization_format to bypass skops UntrustedTypes exception
        mlflow.sklearn.log_model(
            sk_model=model, 
            artifact_path="model_artifact",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        print(f"Metrics: F1={metrics['f1_score']:.4f}, Recall={metrics['recall']:.4f}")
        print("Model and metrics successfully logged to MLflow.\n")
if __name__ == "__main__":
    main()
