import time
import json
import requests
import numpy as np
from dataset import load_data
def main():
    print("Preparing latency benchmark...")
    # Get a sample row from our dataset to use for requests
    X, _ = load_data()
    sample_features = X.iloc[0].tolist()
    url = "http://127.0.0.1:8000/predict"
    payload = {"features": sample_features}
    # Warmup
    try:
        requests.post(url, json=payload)
    except requests.exceptions.ConnectionError:
        print("Error: FastAPI server is not running. Please start it with 'uvicorn src.app:app'")
        return
    latencies = []
    num_requests = 100
    print(f"Sending {num_requests} inference requests...")
    for _ in range(num_requests):
        start_time = time.perf_counter()
        response = requests.post(url, json=payload)
        end_time = time.perf_counter()
        if response.status_code == 200:
            latencies.append((end_time - start_time) * 1000) # Convert to ms
        else:
            print(f"Request failed: {response.text}")
    if not latencies:
        print("No successful requests.")
        return
    p50 = np.percentile(latencies, 50)
    p90 = np.percentile(latencies, 90)
    p99 = np.percentile(latencies, 99)
    report = {
        "total_requests": num_requests,
        "successful_requests": len(latencies),
        "latency_ms": {
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p99": round(p99, 2),
            "mean": round(float(np.mean(latencies)), 2)
        }
    }
    print("\n=== INFERENCE LATENCY REPORT ===")
    print(json.dumps(report, indent=4))
    import os
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/latency_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nLatency report saved to {report_path}")
if __name__ == "__main__":
    main()
