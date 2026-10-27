from src.dataset import get_train_test_split
def test_data_split_no_leakage():
    X_train, X_test, _, _ = get_train_test_split()
    # Ensure no index overlap (data leakage check)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert len(train_indices.intersection(test_indices)) == 0
def test_data_stratification():
    _, _, y_train, y_test = get_train_test_split()
    # Check if class distribution is maintained (difference should be very small)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - test_ratio) < 0.05
