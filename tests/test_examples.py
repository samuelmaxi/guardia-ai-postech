from guardia.examples import EXAMPLES
from guardia.ml.predict import predict, validate

EXPECTED = {"Baixo risco": "low risk", "Risco médio": "mid risk", "Alto risco": "high risk",
            "Alto risco + relato com sinal de atenção": "high risk"}


def test_examples_are_valid_and_predict_as_named():
    for name, ex in EXAMPLES.items():
        assert validate(ex["inputs"]) == []
        assert predict(ex["inputs"])["label"] == EXPECTED[name], name
