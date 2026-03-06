# Datathon - Passos Magicos: Previsao de Risco de Defasagem Escolar

Sistema preditivo de Machine Learning para identificar alunos em risco de defasagem escolar na ONG Passos Magicos. Utiliza dados historicos (PEDE 2022-2024) para emitir alertas antecipados, permitindo intervencao pedagogica a tempo.

## Objetivo

A ONG Passos Magicos precisa identificar rapidamente quais alunos estao em risco de defasagem para intervir a tempo. Este sistema analisa o historico academico do aluno e emite um alerta de risco, priorizando a **sensibilidade (Recall)** para minimizar a chance de deixar um aluno sem suporte.

## Solucao Proposta

- **Modelo:** Random Forest com otimizacao via RandomizedSearchCV
- **Metrica principal:** Recall (~93%) - a cada 100 alunos que realmente precisam de ajuda, o sistema alerta sobre ~93 deles
- **Limiar de decisao:** 0.40 (ao inves de 0.50) - reduz o risco de falsos negativos, priorizando a deteccao de alunos vulneraveis
- **Versionamento de modelos:** MLflow com alias `production` para controle de versoes e rollback

## Arquitetura do Projeto

```
├── app/                          # API FastAPI
│   ├── main.py                   # Inicializacao, logging, middleware, Prometheus
│   ├── routes.py                 # Endpoints: /predict, /reload, /retrain, /
│   └── schemas/
│       ├── aluno_request.py      # Schema de entrada (Pydantic)
│       └── risco_response.py     # Schema de saida (Pydantic)
│
├── src/                          # Pipeline de Machine Learning
│   ├── train.py                  # Pipeline completo de treinamento com MLflow
│   ├── preprocessing.py          # Limpeza e conversao de tipos
│   ├── feature_engineering.py    # Engenharia de features e remocao de data leakage
│   ├── evaluate.py               # Metricas, matriz de confusao, log no MLflow
│   └── utils.py                  # Carga e padronizacao de CSVs multi-ano
│
├── tests/                        # Testes unitarios e de integracao (pytest)
│   ├── conftest.py               # Fixtures e mock global do MLflow
│   ├── test_api.py               # Testes dos endpoints da API
│   ├── test_model.py             # Testes de predicao do modelo
│   ├── test_train.py             # Testes do pipeline de treinamento
│   ├── test_preprocessing.py     # Testes de limpeza de dados
│   ├── test_feature_engineering.py # Testes de engenharia de features
│   ├── test_evaluate.py          # Testes de avaliacao
│   └── test_utils.py             # Testes de carga de dados
│
├── scripts/                      # Scripts utilitarios de MLOps
│   ├── fix_mlflow_db.py          # Correcao de migracoes SQLite do MLflow
│   ├── list_mlflow_models.py     # Listar modelos registrados
│   ├── set_production_alias.py   # Definir alias production em versao do modelo
│   └── xlsx_para_pede2024.py     # Conversao de Excel para CSV
│
├── grafana/                      # Dashboards e datasources do Grafana
│   └── provisioning/
│       ├── datasources/
│       │   └── datasources.yml
│       └── dashboards/
│           ├── dashboards.yml
│           ├── painel.json                         # Dashboard do modelo
│           └── monitoramento_infraestrutura.json   # Dashboard de infra
│
├── notebooks/
│   └── EDA.ipynb                 # Analise Exploratoria de Dados
│
├── files/                        # Dados CSV (PEDE2022, PEDE2023, PEDE2024) - gitignored
├── logs/                         # Logs da API (lidos pelo Promtail) - gitignored
│
├── Dockerfile                    # Imagem Python 3.11-slim com FastAPI
├── docker-compose.yml            # Orquestracao de 7 servicos
├── prometheus.yml                # Configuracao do Prometheus (scrape 5s)
├── promtail-config.yml           # Configuracao do Promtail (log shipping)
└── requirements.txt              # Dependencias Python
```

## Stack de Tecnologia

| Categoria | Tecnologias |
|---|---|
| **Linguagem** | Python 3.11 |
| **ML / Data** | scikit-learn, pandas, numpy, matplotlib |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Model Tracking** | MLflow 3.10.0 (SQLite backend) |
| **Monitoramento** | Prometheus, Grafana, Loki, Promtail, Node Exporter |
| **Testes** | pytest, pytest-cov, httpx |
| **Containerizacao** | Docker, Docker Compose |

## Como Executar

### Pre-requisitos

- Docker e Docker Compose instalados
- Arquivos de dados (`PEDE2022.csv`, `PEDE2023.csv`, `PEDE2024.csv`) na pasta `files/`

### 1. Clonar o repositorio

```bash
git clone <url-do-repositorio>
cd tech_henrique
```

### 2. Subir todos os servicos

```bash
docker compose up --build -d
```

### 3. Acessar os servicos

| Servico | URL | Descricao |
|---|---|---|
| **API (Swagger)** | http://localhost:8000/docs | Documentacao interativa da API |
| **MLflow UI** | http://localhost:5050 | Interface de versionamento de modelos |
| **Grafana** | http://localhost:3000 | Dashboards de monitoramento (senha: admin) |
| **Prometheus** | http://localhost:9090 | Metricas brutas |

### 4. Treinar o modelo (primeira vez)

```bash
# Via endpoint da API (execucao assincrona em background):
curl -X POST http://localhost:8000/retrain

# Ou diretamente no container:
docker exec -it api python src/train.py
```

### 5. Colocar o modelo em producao

Apos o treinamento, acesse o MLflow UI (http://localhost:5050):
1. Navegue ate o modelo registrado `Modelo_Risco_Defasagem`
2. Na versao desejada, adicione o alias `production`
3. Recarregue o modelo na API:

```bash
curl -X POST http://localhost:8000/reload
```

## Endpoints da API

### `GET /` - Health Check
Verifica se a aplicacao esta rodando.
```json
{"status": "ok"}
```

### `POST /predict` - Predicao de Risco
Recebe os dados de um aluno e retorna a probabilidade de risco de defasagem.

**Request:**
```json
{
  "IAA": 7.5,
  "IEG": 8.0,
  "IPS": 6.5,
  "IDA": 7.0,
  "IPV": 6.0,
  "Idade": 15,
  "Fase": "Fase 8",
  "Pedra": "Topazio",
  "Instituicao_de_ensino": "Publica",
  "Genero": "Feminino"
}
```

**Response:**
```json
{
  "risco_defasagem": 0,
  "probabilidade_risco": 0.25,
  "mensagem": "Risco baixo"
}
```

**Campos de entrada:**
| Campo | Tipo | Descricao |
|---|---|---|
| IAA | float | Indice de Autoavaliacao (0-10) |
| IEG | float | Indice de Engajamento (0-10) |
| IPS | float | Indice Psicossocial (0-10) |
| IDA | float | Indice de Aprendizagem (0-10) |
| IPV | float | Indice do Ponto de Virada (0-10) |
| Idade | int | Idade do aluno |
| Fase | str | Fase escolar (ex: "Fase 8", "Alfa") |
| Pedra | str | Classificacao por pedra (ex: "Topazio", "Ametista") |
| Instituicao_de_ensino | str | Tipo de escola ("Publica" ou "Privada") |
| Genero | str | Genero do aluno ("Masculino" ou "Feminino") |

### `POST /retrain` - Retreinamento
Dispara o retreinamento do modelo em background. O progresso pode ser acompanhado nos logs via Grafana/Loki.

### `POST /reload` - Recarregar Modelo
Recarrega o modelo em memoria a partir do MLflow (alias `production`), sem necessidade de reiniciar a aplicacao.

### `GET /metrics` - Metricas Prometheus
Endpoint automatico com metricas de latencia, requisicoes, e metricas customizadas do modelo.

## Pipeline de Machine Learning

### 1. Carga de Dados (`src/utils.py`)
- Carrega CSVs de multiplos anos (2022, 2023, 2024)
- Padroniza nomes de colunas entre anos (ex: `IAA 2022` → `IAA`)
- Unifica em um unico DataFrame

### 2. Pre-processamento (`src/preprocessing.py`)
- Converte formatos numericos (virgula → ponto decimal)
- Trata formato de data do Excel em campo Idade
- Aplica clip em notas (0-10)
- Filtra idades invalidas (<5 ou >30)
- Normaliza colunas categoricas (remove acentos, uppercase)
- Remove nulos da variavel alvo (Defasagem)

### 3. Engenharia de Features (`src/feature_engineering.py`)
- **Variavel alvo:** `Defasagem < 0` → risco = 1, caso contrario = 0
- **Features de interacao:** IEG x IDA (esforco vs resultado), IEG x IAA (esforco vs autoimagem), IPS x IDA (psicologico vs resultado)
- **Conversao de Fase:** texto para numerico (ex: "Fase 8" → 8, "Alfa" → 0)
- **Remocao de Data Leakage:** elimina colunas INDE e IAN (informacao futura que causaria vazamento)
- **Remocao de identificadores:** RA, Nome, Data de Nascimento

### 4. Treinamento (`src/train.py`)
- **Modelo:** RandomForestClassifier dentro de um Pipeline com preprocessamento
- **Tratamento de categoricas:** OneHotEncoder com `handle_unknown='ignore'`
- **Tratamento de nulos:** SimpleImputer (mediana para numericas, moda para categoricas)
- **Otimizacao:** RandomizedSearchCV com 20 iteracoes, 3-fold CV, otimizando para Recall
- **Hiperparametros buscados:** n_estimators, max_depth, min_samples_leaf, max_features, class_weight
- **Registro:** modelo logado automaticamente no MLflow com assinatura e metricas

### 5. Avaliacao (`src/evaluate.py`)
- Metricas: Accuracy, Precision, Recall, F1-Score
- Matriz de confusao salva como artefato no MLflow
- Importancia das features logada no terminal
- Threshold customizado (0.40) aplicado na avaliacao

## Monitoramento

O sistema conta com monitoramento completo via stack Prometheus + Grafana + Loki:

### Metricas coletadas pelo Prometheus
- **Automaticas:** latencia de requisicoes, contagem de requests, status codes (via `prometheus-fastapi-instrumentator`)
- **Customizadas do modelo:**
  - `modelo_predicoes_total` - contador de predicoes por tipo (com/sem risco)
  - `modelo_probabilidade_risco` - histograma de distribuicao de probabilidades
  - `feature_input_iaa` / `feature_input_ieg` - gauge para monitoramento de data drift

### Logs coletados pelo Loki (via Promtail)
- Todas as requisicoes HTTP (IP, metodo, path, status, tempo)
- Eventos de predicao (dados de entrada, risco, probabilidade)
- Eventos de treinamento e recarga de modelo

### Dashboards Grafana
1. **Painel do Modelo** - predicoes, probabilidades, data drift, logs
2. **Monitoramento de Infraestrutura** - CPU, memoria, swap, carga do sistema (via Node Exporter)

## Testes

### Executar todos os testes com cobertura

```bash
python -m pytest -v --cov=app --cov=src tests/
```

### Executar testes do modelo isoladamente

```bash
python -m pytest tests/test_model.py
```

### Estrutura dos testes
- Todos os testes usam mock do MLflow (via `conftest.py`) para evitar poluicao da base
- Testes cobrem: endpoints da API, predicoes, pipeline de treinamento, preprocessamento, engenharia de features, avaliacao e carga de dados

## Metrica de Avaliacao

O modelo e otimizado para **Recall (Sensibilidade)**, pois no contexto da Passos Magicos e preferivel ter falsos positivos (alertar um aluno que nao precisa) do que falsos negativos (deixar de alertar um aluno que precisa de ajuda).

O limiar de decisao foi reduzido de 0.50 para **0.40**, o que significa:
- Probabilidade >= 0.40 → **Risco detectado** (alerta emitido)
- Probabilidade < 0.40 → **Risco baixo**

Essa escolha maximiza a deteccao de alunos vulneraveis, aceitando um trade-off com a precisao.

## Equipe

Henrique Favore Tambolo Helion do Prado Vieira
