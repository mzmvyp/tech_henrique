# Relatório de modificações – Projeto Datathon (repositório do Henrique)

**Objetivo:** Este documento descreve **todas as alterações** feitas no projeto original (repositório Git do Henrique) para que o sistema rode corretamente no Docker, com MLflow funcionando e testes passando. Pode ser enviado ao Henrique para replicar ou incorporar as mudanças no repositório oficial.

**Referência:** Projeto original = repositório do Henrique no Git. Estado atual = cópia local com as modificações descritas abaixo, testada e em funcionamento.

---

## 1. Resumo executivo (tabela)

| # | Arquivo ou item | Tipo | Descrição breve |
|---|------------------|------|-----------------|
| 1 | `prometheus.yml` | Modificado | Target da API alterado de `host.docker.internal:8000` para `api:8000`; adicionado job `machine_resources` para node-exporter. |
| 2 | `docker-compose.yml` | Modificado | Prometheus com `depends_on: api`; MLflow com `working_dir: /app` e `sqlite:////app/mlflow.db`; Loki com `command` explícito. |
| 3 | `docker-compose.portas-separadas.yml` | Novo | Override de portas (API 8001, Grafana 3001, etc.) para rodar junto com outro sistema. |
| 4 | `app/routes.py` | Modificado | Carregamento do modelo **somente via MLflow** (sem fallback para .pkl); nomes de colunas do `/predict` alinhados ao pipeline (`Instituicao_de_ensino`, `Genero`). |
| 5 | `app/schemas/aluno_request.py` | Conferido/igual | Já usa `Instituicao_de_ensino` e `Genero` (nomes que batem com o treino). |
| 6 | `src/train.py` | Modificado | Uso de **apenas os anos cujos CSVs existem** em `files/` (ex.: só 2024 se só existir PEDE2024.csv). |
| 7 | `requirements.txt` | Modificado | MLflow fixado em `mlflow==3.10.0` para evitar diferença de migrações entre host e container. |
| 8 | `scripts/fix_mlflow_db.py` | Novo | Remove tabela temporária `_alembic_tmp_latest_metrics` do SQLite quando a migração do MLflow falha. |
| 9 | `scripts/xlsx_para_pede2024.py` | Novo | Converte o Excel da base PEDE 2024 para `files/PEDE2024.csv` no formato do pipeline. |
| 10 | `scripts/set_production_alias.py` | Novo | Define o alias `production` para uma versão do modelo que tenha artifacts em `mlruns/`. |
| 11 | `scripts/list_mlflow_models.py` | Novo | Lista versões do modelo registrado no MLflow (útil para debug). |
| 12 | `testar_api_completo.py` | Novo | Suíte de testes da API (GET /, /openapi.json, /metrics; POST /predict, /reload, /retrain). |
| 13 | `ADAPTACOES-DOCKER-PARA-HENRIQUE.md` | Novo | Checklist detalhado de adaptações Docker para o Henrique aplicar no Git. |
| 14 | `DOCKER-LEIA-ME.md` | Novo | Instruções de subida no Docker e troubleshooting do MLflow. |
| 15 | Migração do MLflow | Procedimento | Executar `python -m mlflow db upgrade sqlite:///mlflow.db` no host antes de subir a API no Docker (evita erro de migração no container). |
| 16 | Alias `production` | Procedimento | Após treino, garantir que o alias `production` aponte para uma versão cujos artifacts existam em `mlruns/` (ex.: rodar `scripts/set_production_alias.py` se necessário). |

---

## 2. Detalhamento das alterações

### 2.1. `prometheus.yml`

**Problema:** Com o target `host.docker.internal:8000`, o Prometheus dentro do Docker não consegue scrape a API quando ela também está em container.

**Alteração:** Usar o nome do serviço da API na rede do Compose.

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "fastapi_modelo"
    static_configs:
      - targets: ["api:8000"]
  - job_name: "machine_resources"
    static_configs:
      - targets: ["node-exporter:9100"]
```

---

### 2.2. `docker-compose.yml`

- **Prometheus:** adicionar `depends_on: api` para não tentar scrape antes da API estar no ar.
- **MLflow:**
  - `working_dir: /app`
  - No `command`: `--backend-store-uri sqlite:////app/mlflow.db` (caminho absoluto com quatro barras).
  - Evita erros “unable to open database file” e reduz problemas de migração no Windows.
- **Loki:** `command: -config.file=/etc/loki/local-config.yaml` (conferir conforme a imagem usada).

---

### 2.3. `docker-compose.portas-separadas.yml` (novo)

Arquivo de override para usar portas diferentes (ex.: API 8001, Grafana 3001, Prometheus 9091, MLflow 5051, Loki 3101) e não conflitar com outro sistema na mesma máquina.

Uso:
```bash
docker-compose -p datathon -f docker-compose.yml -f docker-compose.portas-separadas.yml up -d --build
```

---

### 2.4. `app/routes.py`

**Alterações:**

1. **Carregamento do modelo apenas via MLflow (sem fallback)**  
   - Removido o fallback que carregava `app/model/modelo.pkl` quando o MLflow falhava.  
   - Agora: se o MLflow falhar ao carregar `models:/Modelo_Risco_Defasagem@production`, a aplicação sobe com `model = None` e `/predict` (e `/reload` em falha) retornam 500. Isso garante que a ML esteja de fato funcionando pelo MLflow.

2. **Remoção do import `joblib`**  
   Não é mais usado.

3. **DataFrame do `/predict`**  
   Os nomes das colunas do DataFrame passado ao modelo devem ser **exatamente** os usados no pipeline de treino. Exemplo:
   - `Instituicao_de_ensino` (e não “Instituição de ensino”).
   - `Genero` (e não “Gênero”).

Trecho relevante do `routes.py` (construção do `df_input`):

```python
data = {
    "IAA": [aluno.IAA],
    "IEG": [aluno.IEG],
    # ...
    "Instituicao_de_ensino": [aluno.Instituicao_de_ensino],
    "Genero": [aluno.Genero]
}
df_input = pd.DataFrame(data)
```

4. **Rota `/reload`**  
   Apenas recarrega o modelo via MLflow; em caso de exceção retorna 500 com a mensagem de erro (sem fallback para .pkl).

---

### 2.5. `src/train.py`

**Alteração:** O treino usa **somente os anos para os quais existem CSV** em `files/`.

Exemplo: se só existir `files/PEDE2024.csv`, o pipeline carrega apenas 2024. Assim o `/retrain` funciona mesmo quando não há PEDE2022 ou PEDE2023.

Trecho (ideia):

```python
paths_base = {
    '2022': 'files/PEDE2022.csv',
    '2023': 'files/PEDE2023.csv',
    '2024': 'files/PEDE2024.csv'
}
paths = {ano: p for ano, p in paths_base.items() if os.path.isfile(p)}
if not paths:
    print("Erro: Nenhum arquivo encontrado em files/ ...")
    return
```

---

### 2.6. `requirements.txt`

**Alteração:** Fixar a versão do MLflow para alinhar host e container e evitar falhas de migração no SQLite.

- De: `mlflow`
- Para: `mlflow==3.10.0`

(Usar a mesma versão com que o banco foi criado/atualizado no host.)

---

### 2.7. Scripts novos (pasta `scripts/`)

| Script | Função |
|--------|--------|
| `fix_mlflow_db.py` | Abre `mlflow.db` e remove a tabela `_alembic_tmp_latest_metrics` (e outras com `_alembic` ou `tmp`), para permitir nova tentativa de migração. Rodar com a API/MLflow parados. |
| `xlsx_para_pede2024.py` | Converte `files/BASE DE DADOS PEDE 2024 - DATATHON.xlsx` em `files/PEDE2024.csv` no formato esperado por `load_data` (nomes de colunas alinhados: Genero, Instituicao_de_ensino, etc.). |
| `set_production_alias.py` | Define o alias `production` do modelo `Modelo_Risco_Defasagem` para uma versão específica (ex.: 5) cujos artifacts existam em `mlruns/`. Ajustar o número da versão no script conforme o ambiente. |
| `list_mlflow_models.py` | Lista as versões do modelo `Modelo_Risco_Defasagem` (version, run_id, source) para debug. |

---

### 2.8. `testar_api_completo.py` (raiz do projeto)

Script que testa:

- GET `/`, `/openapi.json`, `/metrics`
- POST `/predict` (payload válido, inválido, tipos errados, variados Pedra/Fase)
- POST `/reload`
- POST `/retrain`

Uso:
```bash
python testar_api_completo.py --url http://localhost:8000
# ou
python testar_api_completo.py --url http://localhost:8001
```

---

## 3. Procedimentos importantes (não são só “código”)

### 3.1. Migração do banco MLflow (evitar erro no container)

No **host**, na pasta do projeto (onde está o `mlflow.db`):

```bash
python -m mlflow db upgrade sqlite:///mlflow.db
```

Assim o schema do SQLite fica atualizado e o container não tenta rodar uma migração que pode falhar (ex.: “table _alembic_tmp_latest_metrics already exists”).

### 3.2. Alias `production` apontando para versão com artifacts

O modelo é carregado com `models:/Modelo_Risco_Defasagem@production`. Se esse alias apontar para uma versão cujos artifacts não existem em `mlruns/` (ex.: run antigo ou de outra máquina), a API falha ao carregar.

- Após treinar, registrar o modelo e definir o alias `production` para a **versão cuja pasta em `mlruns/` existe** no ambiente atual.
- Pode-se usar `scripts/list_mlflow_models.py` para ver versões e `scripts/set_production_alias.py` para definir o alias (ajustando o número da versão no script).

### 3.3. Dados para treino

- Colocar os CSVs em `files/` (ex.: `PEDE2024.csv`). Se a base for distribuída em Excel, usar `scripts/xlsx_para_pede2024.py` para gerar o CSV no formato correto.
- O pipeline usa apenas os anos para os quais existir arquivo em `files/`.

---

## 4. Documentação criada (para enviar ou commitar)

- **ADAPTACOES-DOCKER-PARA-HENRIQUE.md** – Checklist detalhado de alterações no repositório para Docker (Prometheus, MLflow, Loki, README, .dockerignore, portas separadas, troubleshooting).
- **DOCKER-LEIA-ME.md** – Como subir o projeto no Docker (portas padrão e portas separadas) e como tratar erros comuns do MLflow (`_alembic_tmp_latest_metrics`, `unable to open database file`).
- **RELATORIO-MODIFICACOES-PARA-HENRIQUE.md** – Este arquivo (resumo de tudo que foi modificado em relação ao projeto original).

---

## 5. Ordem sugerida para deixar tudo rodando (pós-clone)

1. Clonar o repositório e aplicar as modificações descritas neste relatório (ou receber um branch/patch com elas).
2. Criar a pasta `files/` e colocar os CSVs (ou gerar a partir do Excel com `scripts/xlsx_para_pede2024.py`).
3. (Opcional) Criar `mlflow.db` vazio na raiz se for rodar MLflow no Docker em ambiente onde o SQLite não cria o arquivo sozinho.
4. No host: `python -m mlflow db upgrade sqlite:///mlflow.db` (se já existir `mlflow.db`).
5. Treinar: `python src/train.py` (ou depois via `POST /retrain` com a API no ar).
6. Garantir que o alias `production` aponte para uma versão com artifacts em `mlruns/` (ex.: rodar `scripts/set_production_alias.py` com a versão correta).
7. Subir os serviços: `docker-compose up -d --build` (ou com `docker-compose.portas-separadas.yml` se usar portas alternativas).
8. Rodar os testes: `python testar_api_completo.py --url http://localhost:8000` (ou 8001).

---

## 6. Observações finais

- **MLflow é obrigatório** para a API: não há mais fallback para `.pkl`. Se o MLflow não carregar o modelo, a API retorna 500 até o problema ser corrigido.
- **Nomes de colunas** na API devem ser os mesmos do pipeline: `Instituicao_de_ensino`, `Genero`, etc. (já garantido em `AlunoRequest` e no `routes.py`).
- **Versão do MLflow** fixada em `3.10.0` para consistência entre host e container.
- Qualquer dúvida sobre um ponto específico pode ser cruzada com os arquivos **ADAPTACOES-DOCKER-PARA-HENRIQUE.md** e **DOCKER-LEIA-ME.md**.

---

**Arquivo:** `RELATORIO-MODIFICACOES-PARA-HENRIQUE.md`  
**Uso:** Enviar ao Henrique junto com os arquivos do projeto (ou referenciar no repositório) para que ele tenha o registro completo do que foi modificado em relação ao projeto original e como reproduzir o ambiente que está rodando agora.
