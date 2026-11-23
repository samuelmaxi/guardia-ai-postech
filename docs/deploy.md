# Deploy na nuvem (manual)

**Responsável:** José · **CI/CD:** manual (sem pipeline automático)

As três opções usam **a mesma imagem Docker** (o `Dockerfile` da raiz). O código não muda de uma nuvem para outra: mudam só os comandos para publicar.

## Antes de começar (vale para todas)

### Requisitos de recurso (medidos)

| Item | Valor medido |
|---|---|
| Tamanho da imagem | ~2,5 GB |
| Tempo de build | ~5 min (treina o modelo e gera o índice do RAG; o índice leva ~4 min) |
| Memória em uso (após a 1ª análise) | **~900 MB**: o modelo de embeddings sozinho ocupa ~460 MB |
| Porta | `8501` (a app também lê `$PORT`) |

> ⚠ **Use pelo menos 1 GB de RAM (recomendado 2 GB).** Os planos Free e Starter do Render (512 MB) **não funcionam**: o container é morto por falta de memória (OOM) na primeira análise.

### Variáveis de ambiente

| Variável | Exemplo | Observação |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `google_genai`, `anthropic`, `xai` ou `groq` |
| `LLM_MODEL` | `gpt-4o-mini` | modelos sugeridos em `.env.example` |
| chave do provedor | `OPENAI_API_KEY=...` | **só a do provedor escolhido**, sempre como *secret* |

Sem essas variáveis a app sobe em **modo degradado**: mostra risco, SHAP e fontes, mas não gera o texto da LLM.

### Teste local da imagem (antes de qualquer nuvem)

```bash
docker build -t guardia-ai .
```

```bash
docker run --rm -p 8501:8501 --env-file .env guardia-ai
```

Abra http://localhost:8501 e rode o exemplo "Alto risco + relato com sinal de atenção".

---

## Opção 1: Azure Container Apps (recomendada: cabe na cota gratuita mensal)

Pré-requisitos: [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) instalada e `az login` feito.

```bash
export RG=guardia-ai-rg LOC=brazilsouth ACR=guardiaai$RANDOM APP=guardia-ai ENV=guardia-env TAG=v1
```

1. Grupo de recursos e registro de imagens:

```bash
az group create -n $RG -l $LOC
```

```bash
az acr create -g $RG -n $ACR --sku Basic --admin-enabled true
```

2. Build da imagem **na própria Azure** (não precisa de Docker local):

```bash
az acr build -r $ACR -t guardia-ai:$TAG .
```

3. Criar a app (1 vCPU / 2 GiB, porta 8501, acesso público):

```bash
az containerapp up -n $APP -g $RG -l $LOC --environment $ENV --image $ACR.azurecr.io/guardia-ai:$TAG --registry-server $ACR.azurecr.io --target-port 8501 --ingress external
```

```bash
az containerapp update -n $APP -g $RG --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 1
```

4. Chave da LLM como secret, e as variáveis apontando para ela:

```bash
az containerapp secret set -n $APP -g $RG --secrets llm-key=COLE_A_CHAVE_AQUI
```

```bash
az containerapp update -n $APP -g $RG --set-env-vars LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini OPENAI_API_KEY=secretref:llm-key
```

5. URL pública:

```bash
az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

**Nova versão (deploy manual):** repita o passo 2 com `TAG=v2` e depois rode:

```bash
az containerapp update -n $APP -g $RG --image $ACR.azurecr.io/guardia-ai:v2
```

**Rollback:** o mesmo comando, apontando para a tag anterior (`:v1`).

**Derrubar tudo depois da avaliação:**

```bash
az group delete -n $RG --yes
```

---

## Opção 2: Render (mais simples, mas exige plano pago)

1. Suba o repositório no GitHub.
2. No [dashboard do Render](https://dashboard.render.com): **New → Web Service →** conecte o repositório. O Render detecta o `Dockerfile` (Runtime: Docker).
3. **Instance type: Standard (2 GB)** ou maior. *Free/Starter (512 MB) não funcionam.*
4. **Environment:** adicione `LLM_PROVIDER`, `LLM_MODEL` e a chave do provedor. Use *Secret Files/Environment* e nunca coloque a chave no repositório.
5. **Settings → Auto-Deploy: Off** (CI/CD manual).
6. Para publicar: **Manual Deploy → Deploy latest commit**. O build leva ~5 min.

**Rollback:** aba **Events** → escolha o deploy anterior → **Rollback**.
**Derrubar:** Settings → **Suspend** (ou Delete) o serviço.

---

## Opção 3: AWS App Runner

Pré-requisitos: [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) com `aws configure` feito e Docker local.

```bash
export REGION=sa-east-1 REPO=guardia-ai TAG=v1
```

```bash
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
```

1. Repositório de imagens (ECR) e push:

```bash
aws ecr create-repository --repository-name $REPO --region $REGION
```

```bash
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
```

```bash
docker build -t $REPO:$TAG . && docker tag $REPO:$TAG $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG && docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:$TAG
```

2. Chave da LLM no Secrets Manager:

```bash
aws secretsmanager create-secret --name guardia-llm-key --secret-string COLE_A_CHAVE_AQUI --region $REGION
```

3. No console do **App Runner → Create service**:
   - Source: *Container registry → Amazon ECR*, imagem `.../guardia-ai:v1`, **Deployment trigger: Manual**.
   - Port: **8501**. CPU/memória: **1 vCPU / 2 GB**.
   - Environment variables: `LLM_PROVIDER`, `LLM_MODEL` e, como *Secrets Manager reference*, a chave do provedor (ex.: `OPENAI_API_KEY` → ARN de `guardia-llm-key`). A *instance role* precisa da permissão `secretsmanager:GetSecretValue` nesse segredo.
   - Health check: TCP na porta 8501.
4. A URL pública aparece em *Default domain*.

> App Runner não está disponível em todas as regiões. Se `sa-east-1` não aparecer no console, use `us-east-1`.

**Nova versão:** faça o push com `TAG=v2`, edite o serviço para apontar para a nova tag e clique em **Deploy**.
**Rollback:** aponte de volta para a tag anterior e clique em **Deploy**.
**Derrubar:** *Actions → Delete service*. Apague também o repositório do ECR e o segredo.

---

## Checklist após o deploy (qualquer nuvem)

- [ ] A URL abre e mostra o banner "IA de apoio".
- [ ] A barra lateral mostra `LLM em uso: provedor:modelo`, não "desligada".
- [ ] O exemplo **Alto risco + relato com sinal de atenção** retorna *Alto risco*, o gráfico SHAP, a explicação da LLM com citações `[arquivo, p. X]` e as fontes (incluindo Ligue 180 / cartilha).
- [ ] A análise aparece na aba **Auditoria**.
- [ ] A aba **Sobre o modelo** mostra a tabela de métricas.

## Limitações conhecidas

- **Log efêmero:** `logs/auditoria.jsonl` fica no disco do container e se perde a cada deploy ou reinício. Para produção seria preciso um volume persistente ou um banco (ex.: Azure Files, EFS, Postgres).
- **Custo:** Azure e AWS cobram fora da cota gratuita, e o Render Standard é pago. **Derrube o serviço depois da avaliação.**
- **Cold start:** com `min-replicas 0` (Azure), a primeira requisição depois de um tempo parado leva alguns segundos.
