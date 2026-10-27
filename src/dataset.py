import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
def load_data():
    """Load Breast Cancer dataset."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target
    return X, y
def get_train_test_split(test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
