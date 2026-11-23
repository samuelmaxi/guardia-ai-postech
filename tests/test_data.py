import pandas as pd

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]


def test_dataset_shape_and_columns():
    df = pd.read_csv("data/maternal_health_risk.csv")
    assert list(df.columns) == FEATURES + ["RiskLevel"]
    assert len(df) == 1014
    assert set(df.RiskLevel) == {"low risk", "mid risk", "high risk"}
