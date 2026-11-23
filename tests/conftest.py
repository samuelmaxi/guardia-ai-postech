import pytest

from guardia.ml.train import MODELS_DIR, load_data, train_and_select


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Garante models/model.joblib para os testes que usam o modelo salvo."""
    if not (MODELS_DIR / "model.joblib").exists():
        train_and_select(load_data())
