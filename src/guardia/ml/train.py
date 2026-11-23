"""Treino e comparação dos modelos de risco materno.

Uso: python -m guardia.ml.train
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = ROOT / "data" / "maternal_health_risk.csv"
MODELS_DIR = ROOT / "models"

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
CLASSES = ["low risk", "mid risk", "high risk"]
TARGET = "RiskLevel"
SEED = 42

RECALL_HIGH = make_scorer(recall_score, labels=["high risk"], average="macro")


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Lê o CSV e remove linhas duplicadas (evita vazamento entre treino e teste)."""
    return pd.read_csv(path).drop_duplicates().reset_index(drop=True)


def make_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED)
        ),
        "hist_gradient_boosting": make_pipeline(HistGradientBoostingClassifier(random_state=SEED)),
    }


def _test_metrics(y_true, y_pred) -> dict:
    return {
        "recall_high": float(recall_score(y_true, y_pred, labels=["high risk"], average="macro")),
        "precision_high": float(precision_score(y_true, y_pred, labels=["high risk"], average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def split(df: pd.DataFrame):
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=SEED
    )


def train_and_select(df: pd.DataFrame, out_dir: Path = MODELS_DIR) -> dict:
    X_train, X_test, y_train, y_test = split(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in make_models().items():
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring={"recall_high": RECALL_HIGH, "f1_macro": "f1_macro"})
        model.fit(X_train, y_train)
        fitted[name] = model
        results[name] = {
            "cv_recall_high": float(cv_res["test_recall_high"].mean()),
            "cv_f1_macro": float(cv_res["test_f1_macro"].mean()),
            "test": _test_metrics(y_test, model.predict(X_test)),
        }
    best = max(results, key=lambda n: (results[n]["cv_recall_high"], results[n]["cv_f1_macro"]))
    metrics = {
        "best": best,
        "n_rows_raw": int(len(pd.read_csv(DATA_PATH))),
        "n_rows_dedup": int(len(df)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": best, "pipeline": fitted[best], "background": X_train}, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def save_figures(df: pd.DataFrame, metrics: dict, out_dir: Path = MODELS_DIR) -> None:
    """Gera as figuras do relatório: matrizes de confusão e SHAP global do vencedor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from sklearn.metrics import ConfusionMatrixDisplay

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, res in metrics["models"].items():
        disp = ConfusionMatrixDisplay(pd.DataFrame(res["test"]["confusion_matrix"]).values, display_labels=CLASSES)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(fig_dir / f"confusion_{name}.png", dpi=120)
        plt.close("all")

    from guardia.ml.predict import _explainer, _bundle

    X_train, *_ = split(df)
    explanation = _explainer()(X_train)
    high = list(_bundle()["pipeline"].classes_).index("high risk")
    shap.plots.beeswarm(explanation[:, :, high], show=False)
    plt.title("SHAP global (classe high risk)")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_global.png", dpi=120)
    plt.close("all")


if __name__ == "__main__":
    data = load_data()
    m = train_and_select(data)
    for n, r in m["models"].items():
        print(f"{n:24s} cv_recall_high={r['cv_recall_high']:.3f} cv_f1={r['cv_f1_macro']:.3f} test_recall_high={r['test']['recall_high']:.3f} test_f1={r['test']['f1_macro']:.3f}")
    print("vencedor:", m["best"])
    save_figures(data, m)
