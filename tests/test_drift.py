from src.drift_check import detect_univariate_drift, generate_synthetic_shift
from src.dataset import load_data
def test_no_drift_detected_on_unchanged_data():
    X, _ = load_data()
    ref_data = X["mean radius"]
    prod_data = ref_data.sample(frac=0.3, random_state=1)
    result = detect_univariate_drift(ref_data, prod_data, "mean radius")
    assert result["drift_detected"] is False
def test_drift_detected_on_synthetically_shifted_data():
    X, _ = load_data()
    ref_data = X["mean radius"]
    prod_data = ref_data.sample(frac=0.3, random_state=1)
    # Apply significant shift
    shifted_prod_data = generate_synthetic_shift(prod_data, shift_factor=1.5)
    result = detect_univariate_drift(ref_data, shifted_prod_data, "mean radius")
    assert result["drift_detected"] is True
