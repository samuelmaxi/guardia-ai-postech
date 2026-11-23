"""Busca de trechos relevantes no índice FAISS."""
from functools import cache

from langchain_community.vectorstores import FAISS

from guardia.rag.ingest import INDEX_DIR, embeddings


@cache
def _store() -> FAISS:
    # o índice é gerado por nós mesmos (ingest.py), por isso a desserialização é segura
    return FAISS.load_local(str(INDEX_DIR), embeddings(), allow_dangerous_deserialization=True)


def retrieve(query: str, k: int = 4) -> list[dict]:
    return [
        {"text": d.page_content, "source": d.metadata["source"], "page": int(d.metadata["page"])}
        for d in _store().similarity_search(query, k=k)
    ]
