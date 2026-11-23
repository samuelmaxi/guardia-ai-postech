from guardia.audit import log_analysis, read_log


def test_log_creates_dir_and_appends(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "nova" / "a.jsonl"))
    i1 = log_analysis({"x": 1})
    i2 = log_analysis({"x": 2, "texto": "pressão"})
    rows = read_log()
    assert [r["id"] for r in rows] == [i2, i1] and "timestamp" in rows[0]
    assert "pressão" in (tmp_path / "nova" / "a.jsonl").read_text()


def test_read_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "nao.jsonl"))
    assert read_log() == []
