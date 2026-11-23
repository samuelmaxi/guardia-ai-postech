"""Guardiã AI: interface Streamlit para a profissional que atende a gestante."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from guardia.audit import log_path, read_log
from guardia.examples import EXAMPLES
from guardia.graph import run_analysis
from guardia.llm import llm_label
from guardia.ml.predict import CLASSES_PT, LABELS_PT, RANGES
from guardia.ml.train import FEATURES, MODELS_DIR
from guardia.prompts import DISCLAIMER

BANNER = "IA de apoio: não substitui a avaliação nem a decisão da profissional."
COLORS = {"low risk": "#2e7d32", "mid risk": "#ed8936", "high risk": "#c62828"}
STEP = {"Age": 1, "SystolicBP": 1, "DiastolicBP": 1, "BS": 0.1, "BodyTemp": 0.1, "HeartRate": 1}

st.set_page_config(page_title="Guardiã AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Guardiã AI")
st.warning(BANNER, icon="⚠️")

with st.sidebar:
    st.markdown("**LLM em uso:** `" + llm_label() + "`")
    st.caption("Configure LLM_PROVIDER, LLM_MODEL e a chave no .env.")
    st.caption("Dados de demonstração sintéticos. Não insira dados reais identificáveis.")

tab_new, tab_audit, tab_model = st.tabs(["Nova análise", "Auditoria", "Sobre o modelo"])


def _load_example():
    st.session_state.result = None  # não deixar o resultado anterior ao lado de entradas novas
    ex = EXAMPLES.get(st.session_state.example)
    if ex:
        for f in FEATURES:
            st.session_state[f] = float(ex["inputs"][f])
        st.session_state.relato = ex["relato"]


with tab_new:
    st.selectbox("Preencher com exemplo sintético", ["—", *EXAMPLES], key="example", on_change=_load_example)
    with st.form("analise"):
        cols = st.columns(3)
        inputs = {}
        for i, f in enumerate(FEATURES):
            lo, hi = RANGES[f]
            st.session_state.setdefault(f, float(EXAMPLES["Baixo risco"]["inputs"][f]))
            inputs[f] = cols[i % 3].number_input(LABELS_PT[f], min_value=float(lo), max_value=float(hi), step=float(STEP[f]), key=f)
        relato = st.text_area("Relato do atendimento (opcional)", key="relato", height=120,
                              help="Texto livre. A IA resume e aponta sinais de atenção; não decide medidas.")
        submitted = st.form_submit_button("Analisar", type="primary")

    if submitted:
        with st.spinner("Analisando: modelo, explicabilidade, documentos e LLM..."):
            try:
                st.session_state.result = run_analysis(inputs, relato)
            except Exception as e:  # o índice ou o modelo podem faltar fora do Docker
                st.error(f"Falha ao executar a análise: {type(e).__name__}: {e}")
                st.session_state.result = None

    s = st.session_state.get("result")
    if s and s.get("errors"):
        st.error("Entrada inválida:\n\n" + "\n".join(f"- {e}" for e in s["errors"]))
    elif s:
        pred = s["prediction"]
        left, right = st.columns([1, 1])
        with left:
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:{COLORS[pred['label']]};color:white'>"
                f"<b>Classificação do modelo:</b> {CLASSES_PT[pred['label']]} "
                f"({pred['probas'][pred['label']]:.0%})</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Modelo: {pred['model_name']}")
            for c, p in pred["probas"].items():
                st.progress(p, text=f"{CLASSES_PT[c]}: {p:.0%}")
        with right:
            st.markdown("**O que pesou nesta predição (SHAP)**")
            shap_df = pd.DataFrame({
                "Fator": [f"{LABELS_PT[x['feature']]} = {x['value']:g}" for x in s["shap_top"]],
                "Impacto": [x["shap"] for x in s["shap_top"]],
            })
            st.altair_chart(
                alt.Chart(shap_df).mark_bar().encode(
                    x=alt.X("Impacto:Q", title="Contribuição SHAP"),
                    y=alt.Y("Fator:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                    color=alt.condition("datum.Impacto > 0", alt.value("#c62828"), alt.value("#2e7d32")),
                ),
                width="stretch",
            )
            st.caption("Positivo: empurra para a classe prevista. Negativo: afasta dela.")

        st.subheader("Explicação para a profissional")
        if s.get("explanation"):
            st.markdown(s["explanation"])
        elif s.get("llm_error"):
            st.error(f"A LLM não respondeu ({s['llm_error']}). O resultado do modelo e as fontes abaixo continuam válidos.")
        else:
            st.info("LLM desligada: configure LLM_PROVIDER, LLM_MODEL e a chave no .env para gerar a explicação em texto.")

        with st.expander(f"Fontes consultadas ({len(s['docs'])} trechos)"):
            for d in s["docs"]:
                st.markdown(f"**{d['source']}, p. {d['page']}**")
                st.caption(d["text"])
        st.warning(DISCLAIMER)
        st.caption(f"Análise registrada na auditoria: `{s['audit_id']}`")

with tab_audit:
    rows = read_log()
    st.markdown(f"**{len(rows)} análises registradas** em `{log_path()}`")
    st.caption("Em deploy na nuvem (Render/Azure/AWS) o disco do container é efêmero: o log é apagado a cada novo deploy.")
    if rows:
        st.dataframe(pd.DataFrame([{
            "quando": r["timestamp"], "status": r["status"], "llm": r["llm"],
            "risco": CLASSES_PT.get((r.get("prediction") or {}).get("label"), "—"),
            "fontes": ", ".join(f"{d['source']} p.{d['page']}" for d in r.get("docs", [])),
            "id": r["id"],
        } for r in rows]), width="stretch", hide_index=True)
        st.download_button("Baixar log (JSONL)", "\n".join(json.dumps(r, ensure_ascii=False) for r in reversed(rows)),
                           file_name="auditoria.jsonl", mime="application/json")

with tab_model:
    metrics_file = MODELS_DIR / "metrics.json"
    if metrics_file.exists():
        m = json.loads(metrics_file.read_text())
        st.markdown(f"Dataset *Maternal Health Risk* (UCI): {m['n_rows_raw']} linhas, **{m['n_rows_dedup']} após remover duplicatas**. Vencedor: **{m['best']}**.")
        st.dataframe(pd.DataFrame({
            n: {"Recall alto risco (CV)": r["cv_recall_high"], "F1 macro (CV)": r["cv_f1_macro"],
                "Recall alto risco (teste)": r["test"]["recall_high"], "Precisão alto risco (teste)": r["test"]["precision_high"],
                "F1 macro (teste)": r["test"]["f1_macro"], "Acurácia (teste)": r["test"]["accuracy"]}
            for n, r in m["models"].items()
        }).T.style.format("{:.3f}"), width="stretch")
        st.markdown(
            "**Por que recall da classe alto risco?** Deixar de sinalizar uma gestante de alto risco (falso negativo) "
            "pode atrasar um cuidado urgente. Um falso positivo gera uma avaliação a mais, custo bem menor. "
            "Por isso o modelo é escolhido pelo maior recall de alto risco na validação cruzada."
        )
    else:
        st.info("Modelo ainda não treinado: rode `python -m guardia.ml.train`.")
