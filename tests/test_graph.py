import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from guardia.audit import read_log
from guardia.graph import run_analysis
from guardia.prompts import DISCLAIMER

HIGH = {"Age": 35, "SystolicBP": 140, "DiastolicBP": 100, "BS": 15.0, "BodyTemp": 98.0, "HeartRate": 80}
LOW = {"Age": 22, "SystolicBP": 100, "DiastolicBP": 70, "BS": 6.9, "BodyTemp": 98.0, "HeartRate": 70}


def fake(query, k=4):
    return [{"text": "trecho", "source": "a.pdf", "page": 1}]


def boom_retriever(query, k=4):
    raise AssertionError("retriever não deveria ser chamado")


class FailingLLM(FakeListChatModel):
    def invoke(self, *a, **k):
        raise RuntimeError("401 chave inválida")


@pytest.fixture(autouse=True)
def audit_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "a.jsonl"))


def test_full_flow():
    s = run_analysis(HIGH, "relato", llm=FakeListChatModel(responses=["texto"]), retriever=fake)
    assert s["prediction"]["label"] == "high risk" and s["explanation"].endswith(DISCLAIMER)
    row = read_log()[0]
    assert row["status"] == "ok" and row["docs"] == [{"source": "a.pdf", "page": 1}]
    assert row["id"] == s["audit_id"] and row["inputs"] == HIGH and row["relato"] == "relato"


def test_invalid_input_skips_everything():
    s = run_analysis({**HIGH, "Age": 999}, llm=FailingLLM(responses=[""]), retriever=boom_retriever)
    assert s["errors"] and "prediction" not in s and read_log()[0]["status"] == "rejeitada"


def test_llm_failure_keeps_ml_and_docs():
    s = run_analysis(HIGH, llm=FailingLLM(responses=[""]), retriever=fake)
    assert s["prediction"] and s["docs"] and s["explanation"] is None and "401" in s["llm_error"]
    assert read_log()[0]["status"] == "erro_llm"


def test_no_llm():
    s = run_analysis(LOW, llm=None, retriever=fake)
    assert s["explanation"] is None and read_log()[0]["status"] == "sem_llm"
