# Passos Mágicos - Previsão de Risco de Defasagem Escolar

## 1) Visão Geral do Projeto

**Objetivo:** Prever o risco de defasagem escolar de estudantes da Associação Passos Mágicos, permitindo intervenção preventiva para crianças e jovens em vulnerabilidade social.

**Solução Proposta:** Pipeline completa de Machine Learning, desde o pré-processamento dos dados até o deploy do modelo em produção via API REST, com monitoramento contínuo de drift e infraestrutura.

**Stack Tecnológica:**
- **Linguagem:** Python 3.11
- **Frameworks de ML:** scikit-learn, pandas, numpy
- **API:** FastAPI + Uvicorn
- **Serialização:** joblib + MLflow Model Registry
- **Testes:** pytest + pytest-cov
- **Empacotamento:** Docker + Docker Compose
- **Deploy:** Local via Docker Compose
- **Monitoramento:** Prometheus + Grafana + Loki/Promtail (logs + métricas + dashboards de drift)

---

## 2) Estrutura do Projeto

```
tech_henrique/
├── app/                          # Aplicação FastAPI
│   ├── main.py                   # Inicialização da API, middleware e logging
│   ├── routes.py                 # Endpoints (/predict, /reload, /retrain)
│   ├── model/                    # Modelo treinado (.pkl) - gerado pelo pipeline
│   └── schemas/
│       ├── aluno_request.py      # Schema de entrada (Pydantic)
│       └── risco_response.py     # Schema de saída (Pydantic)
│
├── src/                          # Pipeline de Machine Learning
│   ├── utils.py                  # Carregamento e unificação dos dados (2022-2024)
│   ├── preprocessing.py          # Limpeza e conversão de tipos
│   ├── feature_engineering.py    # Criação de features, encoding e remoção de leakage
│   ├── train.py                  # Pipeline de treinamento com MLflow
│   └── evaluate.py               # Avaliação e métricas do modelo
│
├── tests/                        # Testes unitários
│   ├── conftest.py               # Configuração e fixtures (mock do MLflow)
│   ├── test_api.py               # Testes dos endpoints da API
│   ├── test_preprocessing.py     # Testes da limpeza de dados
│   ├── test_feature_engineering.py # Testes da engenharia de features
│   ├── test_evaluate.py          # Testes da avaliação do modelo
│   ├── test_train.py             # Testes do pipeline de treinamento
│   ├── test_utils.py             # Testes do carregamento de dados
│   └── test_model.py             # Testes de integração do modelo (.pkl)
│
├── notebooks/
│   └── EDA.ipynb                 # Análise Exploratória de Dados
│
├── grafana/provisioning/         # Dashboards e datasources Grafana
│   ├── dashboards/
│   │   ├── painel.json           # Dashboard principal (drift + predições)
│   │   └── monitoramento_infraestrutura.json
│   └── datasources/
│       └── datasources.yml       # Prometheus + Loki
│
├── files/                        # Dados CSV (não versionados)
│   ├── PEDE2022.csv
│   ├── PEDE2023.csv
│   └── PEDE2024.csv
│
├── Dockerfile                    # Imagem Docker da API
├── docker-compose.yml            # Orquestração dos 7 serviços
├── requirements.txt              # Dependências Python
├── prometheus.yml                # Configuração do Prometheus
└── promtail-config.yml           # Configuração do Promtail (logs)
```

---

## 3) Instruções de Deploy

### Pré-requisitos
- Python 3.11+
- Docker e Docker Compose
- Arquivos CSV na pasta `files/` (PEDE2022.csv, PEDE2023.csv, PEDE2024.csv)

### Instalação Local (sem Docker)

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo
python src/train.py

# Iniciar a API
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Deploy com Docker Compose (recomendado)

```bash
# Construir e iniciar todos os serviços
docker compose up --build -d

# Verificar se os serviços estão rodando
docker compose ps
```

**Serviços disponíveis após o deploy:**

| Serviço        | URL                          | Descrição                      |
|----------------|------------------------------|--------------------------------|
| API FastAPI    | http://localhost:8000        | API de predição                |
| API Docs       | http://localhost:8000/docs   | Documentação Swagger           |
| MLflow UI      | http://localhost:5050        | Registro de modelos            |
| Grafana        | http://localhost:3000        | Dashboards (admin/admin)       |
| Prometheus     | http://localhost:9090        | Métricas                       |

### Comandos Úteis

```bash
# Treinar o modelo (dentro do container)
docker compose exec api python src/train.py

# Rodar testes
docker compose exec api python -m pytest tests/ -v --cov=app --cov=src

# Ver logs
docker compose logs -f api
```

---

## 4) Exemplos de Chamadas à API

### Health Check
```bash
curl http://localhost:8000/
# Resposta: {"status": "ok"}
```

### Predição de Risco (via cURL)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "IAA": 5.5,
    "IEG": 2.0,
    "IPS": 6.0,
    "IDA": 4.5,
    "IPV": 7.0,
    "Idade": 14,
    "Fase": "FASE 3",
    "Pedra": "Quartzo",
    "Instituicao_de_ensino": "Escola Municipal",
    "Genero": "M"
  }'
```

**Resposta esperada (aluno em risco):**
```json
{
  "risco_defasagem": 1,
  "probabilidade_risco": 0.7234,
  "mensagem": "ALERTA: Risco detectado!"
}
```

**Resposta esperada (aluno sem risco):**
```json
{
  "risco_defasagem": 0,
  "probabilidade_risco": 0.1523,
  "mensagem": "Risco baixo"
}
```

### Predição via Python
```python
import httpx

response = httpx.post("http://localhost:8000/predict", json={
    "IAA": 8.5,
    "IEG": 9.0,
    "IPS": 7.5,
    "IDA": 8.0,
    "IPV": 6.5,
    "Idade": 16,
    "Fase": "FASE 8",
    "Pedra": "Topázio",
    "Instituicao_de_ensino": "Escola Estadual",
    "Genero": "F"
})
print(response.json())
```

### Recarregar Modelo (sem reiniciar o servidor)
```bash
curl -X POST http://localhost:8000/reload
# Resposta: {"status": "sucesso", "mensagem": "Modelo atualizado com a última versão de produção!"}
```

### Retreinar Modelo (em background)
```bash
curl -X POST http://localhost:8000/retrain
# Resposta: {"status": "sucesso", "mensagem": "Treinamento iniciado em segundo plano..."}
```

---

## 5) Etapas do Pipeline de Machine Learning

### 5.1 Carregamento dos Dados (`src/utils.py`)
- Lê os CSVs de 2022, 2023 e 2024 (separador `;`)
- Padroniza nomes de colunas que variam entre os anos
- Concatena em um DataFrame unificado com coluna `Ano_Base`

### 5.2 Pré-processamento (`src/preprocessing.py`)
- Converte números do formato brasileiro (vírgula decimal) para formato numérico
- Trata idades em formato Excel (ex: `1/17/00` → `17`)
- Valida faixa de idade (5-30) e clip de notas (0-10)
- Normaliza texto (NFKD → ASCII, uppercase)
- Remove registros sem variável alvo (Defasagem)

### 5.3 Engenharia de Features (`src/feature_engineering.py`)
- **Target:** `alvo_risco = 1` se Defasagem < 0 (aluno atrasado)
- **Features de interação:** IEG×IDA, IEG×IAA, IPS×IDA
- **Encoding categórico:** Pedra (ordinal: Quartzo→1, Ágata→2, Ametista→3, Topázio→4)
- **Extração numérica:** Fase textual → número (ALFA→0, "FASE 8E"→8)
- **Remoção de leakage:** INDE e IAN são removidos pois são índices compostos calculados a partir das mesmas variáveis de entrada

### 5.4 Treinamento e Validação (`src/train.py`)
- **Modelo:** RandomForestClassifier dentro de um Pipeline sklearn
- **Pré-processamento interno:** SimpleImputer (mediana para numéricos, moda para categóricos) + OneHotEncoder
- **Otimização:** RandomizedSearchCV com 20 iterações, 3-fold CV
- **Métrica de otimização:** Recall (prioriza detectar o maior número de alunos em risco)
- **Threshold customizado:** 0.40 (mais sensível que o padrão 0.50)
- **Balanceamento:** `class_weight='balanced'` para lidar com desbalanceamento de classes
- **Split:** 80/20 estratificado por classe
- **Registro:** Modelo salvo via joblib e MLflow Model Registry

### 5.5 Avaliação (`src/evaluate.py`)
- Métricas: Accuracy, Precision, Recall, F1-Score
- Matriz de confusão visual (salva no MLflow)
- Importância das features (ranking)
- Threshold customizável para ajustar sensibilidade

### 5.6 Monitoramento Contínuo
- **Prometheus:** Coleta métricas da API a cada 5s (latência, requests, custom counters)
- **Grafana Dashboards:**
  - Monitoramento de Data Drift (valores de IAA e IEG ao longo do tempo)
  - Volume de predições (Risco vs Sem Risco)
  - Probabilidade média de risco (últimos 5 min)
  - Terminal de logs em tempo real (via Loki)
- **Loki/Promtail:** Agregação e visualização de logs da API

---

## 6) Testes

```bash
# Rodar todos os testes unitários com cobertura
python -m pytest -v --cov=app --cov=src tests/

# Rodar testes de integração do modelo (requer modelo treinado)
python -m pytest tests/test_model.py -v
```
