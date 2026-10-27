import os
import json
import numpy as np
from scipy.stats import ks_2samp
from src.dataset import load_data
def detect_univariate_drift(ref_data, prod_data, feature_name, p_val_threshold=0.05):
    """
    Detects drift using the Kolmogorov-Smirnov test on a single feature.
    Limitation: Only checks univariate distribution shift. Does not prove model degradation.
    """
    stat, p_value = ks_2samp(ref_data, prod_data)
    drift_detected = p_value < p_val_threshold
    return {
        "feature": feature_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "threshold": p_val_threshold,
        "drift_detected": bool(drift_detected),
        "limitation": "Univariate drift is a signal, not a definitive proof of model quality loss."
    }
def generate_synthetic_shift(data, shift_factor=1.3):
    """Synthetically shift the data to simulate distribution drift."""
    return data * shift_factor
def main():
    X, _ = load_data()
    feature_to_check = "mean radius"
    ref_feature_data = X[feature_to_check]
    # Scenario 1: No Shift (Simulating healthy production traffic)
    prod_data_normal = ref_feature_data.sample(frac=0.4, random_state=42)
    report_normal = detect_univariate_drift(ref_feature_data, prod_data_normal, feature_to_check)
    # Scenario 2: Synthetic Shift (Simulating anomalous production traffic)
    prod_data_shifted = generate_synthetic_shift(prod_data_normal, shift_factor=1.3)
    report_shifted = detect_univariate_drift(ref_feature_data, prod_data_shifted, feature_to_check)
    reports = {
        "scenario_no_shift": report_normal,
        "scenario_synthetic_shift": report_shifted
    }
    print("=== DRIFT DETECTION REPORT ===")
    print(json.dumps(reports, indent=4))
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/drift_report.json"
    with open(report_path, "w") as f:
        json.dump(reports, f, indent=4)
    print(f"\nDrift report saved to {report_path}")
if __name__ == "__main__":
    main()
