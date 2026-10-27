import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataset import get_train_test_split
def main():
    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = get_train_test_split()
    print(f"Training Baseline Model on {len(X_train)} samples...")
    # Baseline Model with fixed parameters for reproducibility
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba))
    }
    print("\nMetrics:")
    print(json.dumps(metrics, indent=4))
    # Ensure metrics directory exists
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/baseline.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\nBaseline metrics saved to metrics/baseline.json")
if __name__ == "__main__":
    main()
