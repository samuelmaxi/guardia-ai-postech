from guardia.ml.predict import explain, predict, validate

HIGH = {"Age": 35, "SystolicBP": 140, "DiastolicBP": 100, "BS": 15.0, "BodyTemp": 98.0, "HeartRate": 80}
LOW = {"Age": 22, "SystolicBP": 100, "DiastolicBP": 70, "BS": 6.9, "BodyTemp": 98.0, "HeartRate": 70}


def test_validate_ok():
    assert validate(LOW) == []


def test_validate_out_of_range():
    assert any("Pressão sistólica" in e for e in validate({**LOW, "SystolicBP": 400}))


def test_validate_missing_and_non_numeric():
    errs = validate({k: v for k, v in LOW.items() if k != "Age"} | {"BS": "abc"})
    assert any("Idade" in e for e in errs) and any("Glicemia" in e for e in errs)


def test_predict_high():
    r = predict(HIGH)
    assert r["label"] == "high risk" and abs(sum(r["probas"].values()) - 1) < 1e-6


def test_predict_low():
    assert predict(LOW)["label"] == "low risk"


def test_explain_sorted():
    e = explain(HIGH)
    assert len(e) == 6
    assert [abs(x["shap"]) for x in e] == sorted((abs(x["shap"]) for x in e), reverse=True)
