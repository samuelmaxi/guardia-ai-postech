from guardia.ml.train import load_data, train_and_select


def test_load_data_dedups():
    df = load_data()
    assert not df.duplicated().any() and len(df) < 1014


def test_train_and_select(tmp_path):
    m = train_and_select(load_data(), out_dir=tmp_path)
    assert set(m["models"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    best = m["best"]
    assert m["models"][best]["cv_recall_high"] == max(v["cv_recall_high"] for v in m["models"].values())
    assert (tmp_path / "model.joblib").exists() and (tmp_path / "metrics.json").exists()
    assert m["models"][best]["test"]["recall_high"] >= 0.65  # piso anti-regressão; ver relatório (452 linhas após deduplicação)
