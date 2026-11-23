import pytest

from guardia.llm import get_llm, llm_label


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    # isola os testes de um .env local
    monkeypatch.setattr("guardia.llm.load_dotenv", lambda *a, **k: None)


def test_no_provider_returns_none(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert get_llm() is None and llm_label() == "desligada"


def test_provider_without_key_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert get_llm() is None


def test_unknown_provider_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "xyz")
    monkeypatch.setenv("LLM_MODEL", "m")
    assert get_llm() is None


def test_configured_provider_builds_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert get_llm() is not None and llm_label() == "openai:gpt-4o-mini"
