"""Fluxo principal da Guardiã AI em LangGraph.

validar_entrada -> prever_risco -> explicar_shap -> recuperar_documentos
-> gerar_explicacao -> registrar_auditoria
(entrada inválida: validar_entrada -> registrar_auditoria)
"""
import re
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from guardia.audit import log_analysis
from guardia.llm import get_llm, llm_label
from guardia.ml.predict import CLASSES_PT, LABELS_PT, explain, predict, validate
from guardia.prompts import build_messages, finalize
from guardia.rag.retrieve import retrieve

_DEFAULT = object()

# Termos que indicam possível situação de violência no relato. Só servem para buscar os
# protocolos da rede de atendimento e mostrá-los à profissional; não decidem nada.
SAFETY_TERMS = re.compile(
    r"\b(agress\w*|agred\w*|bat(eu|e|ia|endo)|empurr\w*|espanc\w*|ameaç\w*|ameac\w*|violên\w*|violen\w*"
    r"|machuc\w*|soc(o|ou|ava)|chut\w*|estupr\w*|abus\w*|medo|xing\w*|humilh\w*|proíbe|proibe|control\w*)\b",
    re.IGNORECASE,
)
SAFETY_QUERY = "violência doméstica contra a mulher: como buscar ajuda, Ligue 180 e rede de atendimento"  # distingue "usar LLM do .env" de llm=None (sem LLM)


class GuardiaState(TypedDict, total=False):
    inputs: dict
    relato: str
    errors: list[str]
    prediction: dict
    shap_top: list[dict]
    docs: list[dict]
    explanation: str | None
    llm_error: str | None
    audit_id: str


def _status(s: GuardiaState, llm) -> str:
    if s.get("errors"):
        return "rejeitada"
    if s.get("llm_error"):
        return "erro_llm"
    return "sem_llm" if llm is None else "ok"


def build_graph(llm=_DEFAULT, retriever: Callable = retrieve):
    label = llm_label() if llm is _DEFAULT else (type(llm).__name__ if llm is not None else "desligada")
    if llm is _DEFAULT:
        llm = get_llm()

    def validar_entrada(s: GuardiaState) -> GuardiaState:
        return {"errors": validate(s["inputs"])}

    def prever_risco(s: GuardiaState) -> GuardiaState:
        return {"prediction": predict(s["inputs"])}

    def explicar_shap(s: GuardiaState) -> GuardiaState:
        return {"shap_top": explain(s["inputs"])}

    def recuperar_documentos(s: GuardiaState) -> GuardiaState:
        top3 = ", ".join(f"{LABELS_PT[x['feature']]} {x['value']:g}" for x in s["shap_top"][:3])
        query = f"gestante {CLASSES_PT[s['prediction']['label']]}; fatores: {top3}"
        relato = s.get("relato", "").strip()
        if not relato:
            return {"docs": retriever(query, k=4)}
        # busca separada para o relato: senão os trechos clínicos dominam e a rede de proteção não aparece
        docs = retriever(query, k=3) + retriever(relato[:500], k=2)
        if SAFETY_TERMS.search(relato):
            docs += retriever(SAFETY_QUERY, k=2)
        unique = {(d["source"], d["page"], d["text"]): d for d in docs}
        return {"docs": list(unique.values())}

    def gerar_explicacao(s: GuardiaState) -> GuardiaState:
        if llm is None:
            return {"explanation": None, "llm_error": None}
        try:
            msgs = build_messages(s["prediction"], s["shap_top"], s["docs"], s.get("relato", ""))
            return {"explanation": finalize(llm.invoke(msgs).text), "llm_error": None}
        except Exception as e:  # falha do provedor não pode derrubar a análise
            return {"explanation": None, "llm_error": f"{type(e).__name__}: {e}"}

    def registrar_auditoria(s: GuardiaState) -> GuardiaState:
        audit_id = log_analysis({
            "status": _status(s, llm),
            "llm": label,
            "model_name": s.get("prediction", {}).get("model_name"),
            "inputs": s["inputs"],
            "relato": s.get("relato", ""),
            "errors": s.get("errors", []),
            "prediction": s.get("prediction"),
            "shap_top": s.get("shap_top"),
            "docs": [{"source": d["source"], "page": d["page"]} for d in s.get("docs", [])],
            "explanation": s.get("explanation"),
            "llm_error": s.get("llm_error"),
        })
        return {"audit_id": audit_id}

    g = StateGraph(GuardiaState)
    for node in (validar_entrada, prever_risco, explicar_shap, recuperar_documentos, gerar_explicacao, registrar_auditoria):
        g.add_node(node.__name__, node)
    g.add_edge(START, "validar_entrada")
    g.add_conditional_edges(
        "validar_entrada",
        lambda s: "registrar_auditoria" if s["errors"] else "prever_risco",
        ["registrar_auditoria", "prever_risco"],
    )
    g.add_edge("prever_risco", "explicar_shap")
    g.add_edge("explicar_shap", "recuperar_documentos")
    g.add_edge("recuperar_documentos", "gerar_explicacao")
    g.add_edge("gerar_explicacao", "registrar_auditoria")
    g.add_edge("registrar_auditoria", END)
    return g.compile()


def run_analysis(inputs: dict, relato: str = "", llm=_DEFAULT, retriever: Callable = retrieve) -> GuardiaState:
    try:
        return build_graph(llm, retriever).invoke({"inputs": inputs, "relato": relato})
    except Exception as e:
        # falha fora da LLM (modelo/índice ausente etc.): a tentativa ainda precisa ficar auditada
        log_analysis({"status": "erro", "error": f"{type(e).__name__}: {e}", "inputs": inputs, "relato": relato})
        raise
