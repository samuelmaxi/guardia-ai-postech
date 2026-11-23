"""Predição de risco, validação de entrada e explicação local com SHAP."""
from functools import cache

import joblib
import numpy as np
import pandas as pd
import shap

from guardia.ml.train import CLASSES, FEATURES, MODELS_DIR

# Faixas fisiológicas plausíveis (unidades do dataset: BS em mmol/L, BodyTemp em °F).
RANGES: dict[str, tuple[float, float]] = {
    "Age": (10, 70),
    "SystolicBP": (60, 200),
    "DiastolicBP": (40, 140),
    "BS": (3.0, 25.0),
    "BodyTemp": (95.0, 106.0),
    "HeartRate": (40, 140),
}

LABELS_PT: dict[str, str] = {
    "Age": "Idade",
    "SystolicBP": "Pressão sistólica",
    "DiastolicBP": "Pressão diastólica",
    "BS": "Glicemia (mmol/L)",
    "BodyTemp": "Temperatura (°F)",
    "HeartRate": "Frequência cardíaca",
}

CLASSES_PT = {"low risk": "Baixo risco", "mid risk": "Risco médio", "high risk": "Alto risco"}


def validate(inputs: dict) -> list[str]:
    errors = []
    for f in FEATURES:
        name = LABELS_PT[f]
        if inputs.get(f) in (None, ""):
            errors.append(f"{name}: campo obrigatório.")
            continue
        try:
            v = float(inputs[f])
        except (TypeError, ValueError):
            errors.append(f"{name}: valor não numérico ({inputs[f]!r}).")
            continue
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            errors.append(f"{name}: {v:g} fora da faixa aceita ({lo:g} a {hi:g}).")
    return errors


@cache
def _bundle() -> dict:
    return joblib.load(MODELS_DIR / "model.joblib")


def _frame(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([[float(inputs[f]) for f in FEATURES]], columns=FEATURES)


def predict(inputs: dict) -> dict:
    b = _bundle()
    proba = b["pipeline"].predict_proba(_frame(inputs))[0]
    probas = {c: float(p) for c, p in zip(b["pipeline"].classes_, proba)}
    probas = {c: probas[c] for c in CLASSES}
    return {"label": max(probas, key=probas.get), "probas": probas, "model_name": b["name"]}


@cache
def _explainer():
    """Retorna função X -> shap.Explanation (n, features, classes) para o modelo salvo."""
    b = _bundle()
    steps = b["pipeline"].steps
    est = steps[-1][1]
    pre = b["pipeline"][:-1]  # scaler (LR) ou passthrough vazio (árvores)
    background = pre.transform(b["background"]) if len(steps) > 1 else b["background"]
    if len(steps) > 1:
        exp = shap.LinearExplainer(est, background)
    else:
        exp = shap.TreeExplainer(est)

    def run(X: pd.DataFrame) -> shap.Explanation:
        Xt = pre.transform(X) if len(steps) > 1 else X
        e = exp(Xt)
        return shap.Explanation(e.values, e.base_values, data=X.values, feature_names=FEATURES)

    return run


def explain(inputs: dict, top: int = 6) -> list[dict]:
    X = _frame(inputs)
    label = predict(inputs)["label"]
    idx = list(_bundle()["pipeline"].classes_).index(label)
    values = np.asarray(_explainer()(X).values)[0, :, idx]
    rows = [{"feature": f, "value": float(X.iloc[0][f]), "shap": float(v)} for f, v in zip(FEATURES, values)]
    return sorted(rows, key=lambda r: abs(r["shap"]), reverse=True)[:top]
