"""Ingestão da base de conhecimento: documentos -> chunks -> índice FAISS.

Uso: python -m guardia.rag.ingest
"""
from pathlib import Path

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
KB_DIR = ROOT / "knowledge_base"
INDEX_DIR = ROOT / "index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SKIP = {"FONTES.md"}
# A cartilha tem versão em espanhol a partir da p. 34; indexamos só a parte em português.
MAX_PAGE = {"cartilha_lei_maria_da_penha.pdf": 33}


def embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def load_documents(kb_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.iterdir()):
        if path.name in SKIP:
            continue
        if path.suffix.lower() == ".pdf":
            pages = PdfReader(path).pages[: MAX_PAGE.get(path.name)]
            for i, page in enumerate(pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append(Document(text, metadata={"source": path.name, "page": i}))
        elif path.suffix.lower() in {".md", ".txt"}:
            docs.append(Document(path.read_text(), metadata={"source": path.name, "page": 1}))
    return docs


def build_index(kb_dir: Path = KB_DIR, out_dir: Path = INDEX_DIR) -> int:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(load_documents(kb_dir))
    FAISS.from_documents(chunks, embeddings()).save_local(str(out_dir))
    return len(chunks)


if __name__ == "__main__":
    print("chunks indexados:", build_index())
