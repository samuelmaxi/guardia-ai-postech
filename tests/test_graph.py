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


def test_relato_gets_its_own_retrieval():
    calls = []

    def spy(query, k=4):
        calls.append((query, k))
        return [{"text": query, "source": f"{len(calls)}.pdf", "page": 1}]

    s = run_analysis(HIGH, "dor de cabeça forte e visão turva", llm=None, retriever=spy)
    assert ("dor de cabeça forte e visão turva", 2) in calls
    assert [d["source"] for d in s["docs"]] == ["1.pdf", "2.pdf"]


def test_no_relato_single_retrieval():
    calls = []
    run_analysis(HIGH, "", llm=None, retriever=lambda q, k=4: calls.append(k) or [])
    assert calls == [4]


def test_attention_terms_trigger_safety_retrieval():
    from guardia.graph import SAFETY_QUERY

    calls = []
    run_analysis(HIGH, "Disse que o marido a empurrou e ela tem medo de voltar.", llm=None,
                 retriever=lambda q, k=4: calls.append(q) or [])
    assert SAFETY_QUERY in calls


def test_benign_relato_does_not_trigger_safety_retrieval():
    from guardia.graph import SAFETY_QUERY

    calls = []
    run_analysis(HIGH, "Refere enjoo pela manhã e cansaço.", llm=None, retriever=lambda q, k=4: calls.append(q) or [])
    assert SAFETY_QUERY not in calls


class BlocksLLM(FakeListChatModel):
    """Simula Gemini 3 / modelos com thinking: content vem como lista de blocos."""

    def invoke(self, *a, **k):
        from langchain_core.messages import AIMessage
        return AIMessage(content=[{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "explicação"}])


def test_llm_content_blocks_are_handled():
    s = run_analysis(HIGH, llm=BlocksLLM(responses=[""]), retriever=fake)
    assert s["llm_error"] is None and s["explanation"].startswith("explicação")


def test_failure_outside_llm_is_still_audited():
    def broken_retriever(query, k=4):
        raise OSError("índice ausente")

    with pytest.raises(OSError):
        run_analysis(HIGH, llm=None, retriever=broken_retriever)
    row = read_log()[0]
    assert row["status"] == "erro" and "índice ausente" in row["error"] and row["inputs"] == HIGH
