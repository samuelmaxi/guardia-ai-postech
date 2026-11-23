import guardia.rag.retrieve as retrieve_mod
from guardia.rag.ingest import build_index
from guardia.rag.retrieve import retrieve


def test_build_and_retrieve(tmp_path, monkeypatch):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "pa.md").write_text("Pressão arterial maior ou igual a 140/90 mmHg na gestação indica hipertensão e exige avaliação.")
    (kb / "violencia.md").write_text("O Ligue 180 é a Central de Atendimento à Mulher em situação de violência.")
    assert build_index(kb, tmp_path / "idx") >= 2
    monkeypatch.setattr(retrieve_mod, "INDEX_DIR", tmp_path / "idx")
    retrieve_mod._store.cache_clear()
    hits = retrieve("telefone para mulher em situação de violência", k=1)
    assert hits[0]["source"] == "violencia.md" and hits[0]["page"] == 1
    retrieve_mod._store.cache_clear()
