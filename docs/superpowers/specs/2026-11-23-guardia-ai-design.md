# Guardiã AI: design da entrega (Tech Challenge Fase 5)

Data: 2026-11-23 · Grupo: Samuel, José, Jeff, Leo

## 1. Intenção

Entregar o Tech Challenge Fase 5 completo, que vale 90% da nota da fase. O critério do enunciado é uma solução **coerente, funcional e responsável**, não a quantidade de tecnologia. Por isso tudo gira em torno de uma jornada ponta a ponta que precisa funcionar ao vivo no vídeo e no deploy.

Fontes de verdade (na pasta-mãe `guardia-ai/`): `docs/enunciado-original.txt` prevalece, `specs/requisitos.md` é o checklist (RF/RNF/ENT).

## 2. Objetivos, não-objetivos, anti-objetivos

**Objetivos**
- Cumprir todos os itens [O] de `specs/requisitos.md` e o desejável RF11 (citar a fonte).
- Uma profissional preenche os sinais vitais e o relato de uma gestante e recebe: classe de risco + probabilidades, o que pesou (SHAP), uma explicação da LLM fundamentada em documentos com fonte citada, o aviso de que a decisão é dela, e a análise registrada em log.
- App publicada na nuvem (Render por padrão, com Azure e AWS como opções), com deploy manual feito pelo José.
- Roteiro do vídeo para 4 pessoas, com o Samuel falando menos.

**Não-objetivos**
- Múltiplos agentes (RF14) e entrada multimodal (RF15). São opcionais e ficam fora.
- CI/CD automatizado. O deploy é manual.
- Login, multiusuário e banco de dados.

**Anti-objetivos** (a entrega falha se fizer isto, mesmo funcionando)
- A LLM dar diagnóstico, citar medicamento ou dose, ou decidir uma medida de segurança.
- A LLM responder com algo que não está no resultado do modelo nem nos trechos recuperados, ou fazer isso sem citar a fonte.
- Uma análise rodar sem ficar registrada no log.
- Usar dado real identificável.

## 3. Restrições

- Cenário: **Opção A**, risco materno, com dataset *Maternal Health Risk* (UCI id 863, CC BY 4.0, 1.014 linhas, 6 features → `low/mid/high risk`).
- Orquestração: **LangGraph** (obrigatório ter LangChain ou LangGraph).
- LLM: **na nuvem, escolhida pelo `.env`**. Precisa aceitar OpenAI, Google Gemini, Anthropic Claude, xAI Grok e Groq.
- Entregáveis: repositório Git, app executável, Dockerfile, relatório PDF, vídeo ≤ 15 min.
- Repositório: `guardia-ai/guardia-ai-entrega/`, com git próprio.

## 4. Abordagens consideradas

| # | Abordagem | Trade-off | Decisão |
|---|---|---|---|
| 1 | **Streamlit único com LangGraph dentro, deploy no Render** | Um container, deploy de um clique, jornada direta no vídeo. Não tem API separada, o que o PDF não exige. | **Escolhida** |
| 2 | FastAPI + Streamlit via docker-compose | Arquitetura mais "profissional", mas são 2 serviços para o José publicar e manter, e o PDF não premia quantidade de tecnologia. | Descartada |
| 3 | Streamlit no Hugging Face Spaces | Grátis, mas o deploy é por git push (menos manual) e o espaço dorme e demora a acordar durante a avaliação. | Descartada |

Escolhida pelo grupo: **1**, porque cumpre todo o obrigatório com o menor número de peças que podem quebrar.

## 5. O que foi decidido (pelo grupo)

- Repositório em pasta própria dentro de `guardia-ai/`.
- Opção A, com **recall da classe `high risk`** como métrica central.
- LLM na nuvem, com provedor dinâmico.
- Deploy na nuvem com CI/CD manual, sob responsabilidade do **José**.
- Vídeo dividido entre Samuel, José, Jeff e Leo, com menos fala para o Samuel.
- Abordagem 1 (Streamlit + Render).

## 6. Design

### 6.1 Estrutura do repositório

```
guardia-ai-entrega/
├── data/maternal_health_risk.csv     # dataset versionado (pequeno, licença CC BY 4.0)
├── notebooks/01_eda_e_modelos.ipynb  # EDA + comparação, roda o mesmo código de src/
├── src/guardia/
│   ├── ml/train.py                   # treina, compara, salva modelo + métricas
│   ├── ml/predict.py                 # carrega modelo, prediz, SHAP
│   ├── rag/ingest.py                 # PDFs/textos → chunks → índice FAISS
│   ├── rag/retrieve.py
│   ├── llm.py                        # fábrica de LLM a partir do .env
│   ├── graph.py                      # LangGraph: o fluxo principal
│   ├── prompts.py                    # prompt com guardrails
│   └── audit.py                      # log JSONL
├── knowledge_base/                   # documentos públicos + FONTES.md (origem/licença de cada um)
├── models/                           # modelo treinado + metrics.json (gerados)
├── app.py                            # Streamlit
├── tests/
├── Dockerfile
├── .env.example
├── docs/deploy.md                    # passo a passo do José (Render, Azure, AWS)
├── docs/roteiro-video.md
├── relatorio/relatorio.md → relatorio.pdf
└── README.md
```

### 6.2 Dados e ML (RF02–RF07)

- **EDA**: distribuições, correlação, balanceamento das classes e duplicatas. Esse dataset tem centenas de linhas duplicadas. **Deduplicar antes do split** evita vazamento entre treino e teste e infla menos as métricas. Isso vai no relatório como decisão de preparação.
- Split estratificado treino/teste (80/20) + validação cruzada estratificada de 5 folds no treino.
- **Modelos (3)**: Regressão Logística (baseline interpretável, com scaler), Random Forest e HistGradientBoosting (gradient boosting do próprio sklearn, sem dependência extra).
- **Métricas**: recall da classe `high risk` (central), F1 macro, precision por classe, matriz de confusão e accuracy só como referência. O relatório discute o custo do erro: um falso negativo em `high` (gestante de alto risco tratada como baixa) é o erro mais caro. Um falso positivo gera uma consulta a mais, o que é aceitável.
- **Seleção**: o melhor recall `high` no CV, com desempate por F1 macro. O vencedor vai para `models/`.
- **Explicabilidade**: SHAP (TreeExplainer para árvores, LinearExplainer para LR) na predição atual, mostrando top features com o sentido do efeito. Também importância global no notebook/relatório.
- Validação de entrada: faixas fisiológicas plausíveis (ex.: idade 10–70, sistólica 60–200). Fora da faixa, a análise não roda e aparece uma mensagem.

### 6.3 Relato (ponte com o contexto de segurança)

O formulário tem um campo de **relato livre** e opcional. O ML não usa o relato. A LLM o resume e aponta **sinais que merecem atenção** (incluindo possíveis indícios de violência), recomendando que a profissional **considere** a rede de atendimento (ex.: Ligue 180) com base nos documentos. Ela nunca decide uma medida de segurança. Os relatos de exemplo da demo são sintéticos.

### 6.4 RAG (RF09–RF11)

- Base: documentos públicos do Ministério da Saúde (gestação de alto risco, pré-natal), Lei 11.340/2006 (Maria da Penha) e material público da rede de atendimento/Ligue 180. A origem e a licença de cada arquivo ficam em `knowledge_base/FONTES.md`.
- Chunks de ~800 caracteres com sobreposição, com metadado `arquivo + página`.
- **Embeddings locais com `fastembed`** (modelo multilíngue pequeno, CPU). Assim o RAG funciona com qualquer provedor de LLM, inclusive os que não têm embeddings (Groq, xAI).
- Índice **FAISS**, gerado no `docker build`, para que o container suba pronto.
- Consulta = classe prevista + top features + resumo do relato. Recupera os top-k=4 trechos. A UI mostra cada trecho com arquivo e página.

### 6.5 LLM dinâmica

`.env`: `LLM_PROVIDER` (`openai|google_genai|anthropic|xai|groq`), `LLM_MODEL` e a chave do provedor. `llm.py` usa `init_chat_model` do LangChain. Adicionar outro provedor suportado pelo LangChain é só instalar o pacote e trocar o `.env`.
**Sem chave configurada**, a app continua rodando: mostra ML + SHAP + trechos do RAG e avisa que a explicação em texto está desligada. Isso protege a demo e quem for avaliar sem chave.

### 6.6 Fluxo LangGraph (RF12)

Grafo linear com estado tipado:

`validar_entrada → prever_risco → explicar_shap → recuperar_documentos → gerar_explicacao → registrar_auditoria`

### 6.7 Guardrails (RNF02–RNF05)

Prompt de sistema: o papel é apoiar a profissional, nunca falar com a paciente. Usar só o resultado do modelo e os trechos recuperados, citando `[arquivo, p. X]`. Se faltar informação, dizer isso. Proibido diagnóstico definitivo, medicamento/dose e decisão de segurança. Sempre terminar lembrando que a decisão é da profissional.
Como camada extra, uma checagem simples na saída: o aviso final é sempre anexado pelo código, não depende da LLM. O banner "IA de apoio, não substitui decisão profissional" fica fixo na UI.

### 6.8 Auditoria (RNF06)

Cada análise grava uma linha em `logs/auditoria.jsonl` com: id, timestamp, provedor/modelo da LLM, versão do modelo ML, entradas, predição + probabilidades, top SHAP, trechos recuperados (arquivo/página) e texto da LLM. Uma aba "Auditoria" no Streamlit lista e permite baixar o arquivo.
Limitação declarada: no Render grátis o disco é efêmero, então o log se perde a cada novo deploy. Isso vai no relatório, com a alternativa de usar banco/disco persistente.

### 6.9 Interface

```
┌ Guardiã AI ─────────────── [IA de apoio · decisão é da profissional] ┐
│ Aba: Nova análise | Auditoria | Sobre o modelo                       │
│ Idade [ ] Sistólica [ ] Diastólica [ ] Glicemia [ ] Temp [ ] FC [ ]  │
│ Relato (opcional) [..............................]  [Exemplos ▼]    │
│ [Analisar]                                                           │
│ ── Resultado ──                                                      │
│ Risco: ALTO (0.87)  ▓▓▓▓▓▓▓▓░░   | gráfico SHAP (o que pesou)        │
│ Explicação da LLM ... [Manual Gestação Alto Risco, p. 42]            │
│ ▸ Fontes consultadas (4 trechos)                                     │
│ ⚠ Esta análise não é diagnóstico. A decisão é da profissional.       │
└──────────────────────────────────────────────────────────────────────┘
```

"Exemplos" preenche casos sintéticos prontos (baixo, médio, alto + relato com sinal de atenção) para o vídeo.

### 6.10 Docker e deploy

- Dockerfile com `python:3.12-slim`: instala dependências, treina o modelo (rápido) e gera o índice no build, depois roda `streamlit` na porta `$PORT`.
- `docs/deploy.md` para o **José**, com três opções que usam a mesma imagem Docker (o código não muda entre elas):
  - **Render (padrão)**: Web Service Docker apontando para o repositório GitHub, env vars (`LLM_PROVIDER`, `LLM_MODEL` e a chave), auto-deploy desligado, publicação por "Manual Deploy".
  - **Azure**: build e push da imagem para o Azure Container Registry, depois **Azure Container Apps** (ou App Service for Containers) com as mesmas env vars, a chave guardada como secret. Comandos `az` passo a passo.
  - **AWS**: build e push para o **ECR**, depois **App Runner** a partir da imagem, com as env vars e a chave no Secrets Manager. Comandos `aws` passo a passo.
  - Em todas: checklist de verificação após o deploy, rollback (voltar para a imagem/tag anterior) e aviso de custo (Azure e AWS cobram fora do free tier, então derrubar o serviço depois da avaliação).
- A app lê a porta de `$PORT` (padrão 8501), o que serve para as três plataformas.

### 6.11 Testes

pytest curto: o grafo roda de ponta a ponta com uma LLM falsa, uma entrada inválida é barrada, uma análise gera uma linha de auditoria com todos os campos, o aviso final sempre aparece, e o modelo salvo atinge o recall `high` mínimo registrado em `metrics.json`.

### 6.12 Relatório e vídeo

- `relatorio/relatorio.md` com problema, dados, preparação, modelos, métricas e custo do erro, LLM, RAG, guardrails, auditoria, deploy e limitações. Convertido para PDF pelo repositório, com métricas e figuras reais geradas pelo treino.
- `docs/roteiro-video.md` (≤ 15 min), em blocos com tempo e falas:

| Pessoa | Bloco | Tempo aprox. |
|---|---|---|
| **Samuel** | Abertura: problema, cenário e aviso de responsabilidade | ~1,5 min |
| **Jeff** | Dados, EDA, modelos, métricas, por que recall `high`, SHAP | ~4 min |
| **Leo** | LLM dinâmica, RAG, guardrails, **demo da jornada completa** | ~5 min |
| **José** | Arquitetura/LangGraph, auditoria, Docker, deploy no Render, limitações e encerramento | ~4 min |

## 7. Chamadas feitas por mim (podem corrigir)

- Nome da pasta: `guardia-ai-entrega/`.
- 3 modelos em vez de 2. HistGradientBoosting no lugar de XGBoost, para não adicionar dependência.
- Deduplicar o dataset antes do split.
- Embeddings locais (`fastembed`) + FAISS.
- App funciona sem chave de LLM (modo degradado).
- Log em JSONL, sem banco.
- Distribuição dos blocos do vídeo e tempos acima.
- Relatório escrito em Markdown e convertido para PDF.

## 8. Deixado para quem implementar

- Modelo de embeddings exato do fastembed e tamanho dos chunks (ajustar vendo a qualidade da recuperação).
- Modelo padrão sugerido por provedor no `.env.example`.
- Faixas exatas da validação de entrada.
- Escolha dos PDFs da base: precisam ser públicos, com a origem registrada em `FONTES.md`, e o download é confirmado com o grupo antes.
- Ferramenta de conversão Markdown → PDF.
