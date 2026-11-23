import pytest
from streamlit.testing.v1 import AppTest

from guardia.rag.ingest import INDEX_DIR


@pytest.mark.skipif(not (INDEX_DIR / "index.faiss").exists(), reason="índice não gerado")
def test_switching_example_clears_previous_result(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "a.jsonl"))
    monkeypatch.setenv("LLM_PROVIDER", "")
    at = AppTest.from_file("../app.py", default_timeout=120).run()
    at.selectbox(key="example").select("Alto risco").run()
    at.button[0].click().run()
    assert any("Alto risco" in m.value for m in at.markdown)
    at.selectbox(key="example").select("Baixo risco").run()
    assert not any("Classificação do modelo" in m.value for m in at.markdown)
