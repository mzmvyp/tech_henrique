# Relatório de Revisão e Correções de Bugs

**Data:** 03/03/2026  
**Contexto:** Revisão minuciosa de todo o projeto após as modificações descritas em `RELATORIO-MODIFICACOES-PARA-HENRIQUE.md`. Foram analisados todos os arquivos do projeto, identificados bugs e aplicadas correções. Este relatório pode ser enviado ao Henrique junto com os arquivos alterados.

---

## 1. Resumo das correções aplicadas

| # | Arquivo | Severidade | Descrição |
|---|---------|------------|-----------|
| 1 | `app/routes.py` | **CRÍTICO** | Textos categóricos (Pedra, Genero, Instituicao_de_ensino) não eram normalizados antes da predição, causando mismatch com o modelo treinado. |
| 2 | `app/routes.py` | **CRÍTICO** | Coluna `Fase` era passada como string ao modelo, mas no treino o pandas a leu como numérica. Causava erro 500 para Fase="Alfa". |
| 3 | `app/routes.py` | **MÉDIO** | Coluna `Pedra_Num` era criada na predição mas NÃO existe no pipeline de treino (`create_features`). |
| 4 | `app/routes.py` | **MENOR** | Import `re` não utilizado foi substituído por `unicodedata` (necessário para a normalização). |
| 5 | `tests/test_train.py` | **MÉDIO** | Testes `test_run_training_pipeline` e `test_run_training_file_error` falhavam quando os CSVs não existiam no disco. |
| 6 | `tests/test_model.py` | **MENOR** | Dados de teste incluíam `Pedra_Num` (que o modelo não espera) e `Instituicao_de_ensino` em minúsculas. |
| 7 | `.gitignore` | **MENOR** | Caractere espúrio `ß` na linha `logs/ß` impedia o diretório `logs/` de ser ignorado corretamente. |

---

## 2. Detalhamento das correções

### 2.1. `app/routes.py` — Normalização de texto (BUG CRÍTICO)

**Problema:** Durante o treino, a função `clean_data()` em `preprocessing.py` normaliza TODAS as colunas de texto para **MAIÚSCULAS SEM ACENTOS** (via NFKD + encode ASCII + upper). O `OneHotEncoder` do modelo sklearn é treinado com esses valores normalizados (ex: `"AGATA"`, `"QUARTZO"`, `"TOPAZIO"`, `"F"`, `"ESCOLA PUBLICA"`).

No endpoint `/predict`, os valores categóricos eram passados diretamente ao modelo **sem nenhuma normalização**. Isso significava que:
- Usuário envia `"Pedra": "Ágata"` → modelo recebe `"Ágata"` → `OneHotEncoder(handle_unknown='ignore')` gera vetor de zeros → **informação da Pedra é completamente perdida**
- Mesmo `"Pedra": "Agata"` (sem acento, mas minúsculo) → não bate com `"AGATA"` do treino

**Impacto:** A informação de TODAS as variáveis categóricas (Pedra, Genero, Fase, Instituicao_de_ensino) era silenciosamente descartada pelo modelo em produção, degradando a qualidade das previsões.

**Correção:**
```python
import unicodedata

def _normalizar_texto(valor):
    """Aplica a mesma normalização que clean_data usa no treino."""
    s = str(valor).strip()
    s = unicodedata.normalize('NFKD', s).encode('ascii', errors='ignore').decode('utf-8')
    return s.upper()
```

Aplicada a cada campo categórico antes de montar o DataFrame:
```python
data = {
    ...
    "Fase": [_normalizar_texto(aluno.Fase)],
    "Pedra": [_normalizar_texto(aluno.Pedra)],
    "Instituicao_de_ensino": [_normalizar_texto(aluno.Instituicao_de_ensino)],
    "Genero": [_normalizar_texto(aluno.Genero)]
}
```

### 2.2. `app/routes.py` — Coluna `Fase` como string causava erro 500 (BUG CRÍTICO)

**Problema:** No dado de treino (PEDE2024.csv), a coluna `Fase` continha apenas valores numéricos (1, 2, 3, ..., 8). O pandas leu como `int64`, e o `ColumnTransformer` com `make_column_selector(dtype_exclude="object")` a roteou para o transformador numérico (`SimpleImputer(strategy='median')`).

No endpoint `/predict`, `Fase` era passada como string (ex: `"8"`, `"Alfa"`). Para `"8"`, o SimpleImputer conseguia converter para float. Para `"Alfa"`, falhava com: `"Cannot use median strategy with non-numeric data: could not convert string to float: 'ALFA'"`.

**Impacto:** Qualquer predição com `Fase="Alfa"`, `"Alpha"` ou outro texto não-numérico retornava **erro 500**.

**Correção:** A coluna `Fase` agora é convertida para seu valor numérico (`extrair_fase`) antes de ser inserida no DataFrame, garantindo que tenha o mesmo dtype numérico que o modelo espera:

```python
fase_num = extrair_fase(_normalizar_texto(aluno.Fase))
data = {
    ...
    "Fase": [fase_num],  # numérico, como no treino
    ...
}
df_input['Fase_Num'] = [fase_num]
```

### 2.4. `app/routes.py` — Remoção de `Pedra_Num` (BUG MÉDIO)

**Problema:** O endpoint `/predict` criava uma coluna `Pedra_Num` (mapeamento numérico de Pedra), porém essa coluna **NÃO é criada** no pipeline de treino (`src/feature_engineering.py` → `create_features`). O pipeline de treino cria apenas `Fase_Num`, não `Pedra_Num`.

O modelo sklearn (`ColumnTransformer` com `make_column_selector`) ignorava a coluna extra silenciosamente, mas:
- O código era enganoso (sugeria que o modelo usa `Pedra_Num`)
- O `pedra_map` usava valores com acento (`Ágata`, `Topázio`) que nunca batiam com os dados normalizados

**Correção:** Removidas as linhas:
```python
# REMOVIDO:
pedra_map = {'Quartzo': 1, 'Ágata': 2, 'Ametista': 3, 'Topázio': 4}
df_input['Pedra_Num'] = df_input['Pedra'].map(pedra_map).fillna(1)
```

### 2.5. `app/routes.py` — Import `re` não utilizado

**Problema:** O `import re` na linha 6 não era usado em nenhum lugar do arquivo.

**Correção:** Substituído por `import unicodedata` (necessário para a normalização de texto adicionada).

### 2.6. `tests/test_train.py` — Testes falhando sem CSVs (BUG MÉDIO)

**Problema:** A função `run_training()` em `train.py` foi modificada para verificar a existência dos CSVs via `os.path.isfile()` antes de processá-los:
```python
paths = {ano: p for ano, p in paths_base.items() if os.path.isfile(p)}
if not paths:
    return  # Sai cedo sem chamar load_data
```

Os testes `test_run_training_pipeline` e `test_run_training_file_error` **não mockavam** `os.path.isfile`, então quando os CSVs não existiam no disco (ambiente CI, outra máquina), `paths` ficava vazio, a função retornava imediatamente, e `mock_load.assert_called_once()` falhava.

**Correção:** Adicionado `@patch('os.path.isfile', return_value=True)` a ambos os testes:
```python
@patch('os.path.isfile', return_value=True)
def test_run_training_pipeline(mock_isfile, ...):
    ...

@patch('os.path.isfile', return_value=True)
def test_run_training_file_error(mock_isfile, ...):
    ...
```

### 2.7. `tests/test_model.py` — Dados de teste inconsistentes

**Problema:** O DataFrame de teste incluía `Pedra_Num` (que o modelo não espera) e `Instituicao_de_ensino` com valor `"Escola Publica"` (minúsculas misturadas) em vez de `"ESCOLA PUBLICA"` (como o modelo foi treinado).

**Correção:** Removida `Pedra_Num` e corrigido `Instituicao_de_ensino` para `"ESCOLA PUBLICA"`.

### 2.8. `.gitignore` — Caractere espúrio `ß`

**Problema:** A linha 48 continha `logs/ß` (com caractere alemão ß), que é um path inválido. O diretório `logs/` não era corretamente ignorado pelo Git.

**Correção:** Alterado para `logs/`.

---

## 3. Observações adicionais (não corrigidos — avaliar futuramente)

### 3.1. `src/train.py` ainda salva `.pkl` (inconsistência com design "MLflow only")

O treino ainda executa:
```python
joblib.dump(best_model, 'app/model/modelo.pkl')
```
Isso contradiz o design onde a API usa APENAS MLflow. O `.pkl` não é mais usado pela API. Sugestão: remover essas linhas e o `import joblib` do train.py após confirmar que tudo funciona 100% via MLflow.

### 3.2. Alias `production` não é definido automaticamente após treino

Após `/retrain`, o modelo é registrado no MLflow mas o alias `production` NÃO é definido automaticamente. O fluxo atual exige intervenção manual (`scripts/set_production_alias.py`). Sugestão futura: automatizar isso no final de `train.py`.

### 3.3. `preprocessing.py` — Potencial corrupção de números com ponto decimal

A função `clean_data` faz:
```python
df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.')
```

Isso assume formato brasileiro (`.` = milhar, `,` = decimal). Se algum valor já usar ponto como decimal (ex: `7.0`), ele é corrompido: `7.0` → `70` → `70.0` (depois clipado a `10.0`). Na prática, se os CSVs usam consistentemente formato brasileiro, isso não ocorre. Mas é um risco se os dados forem editados manualmente ou exportados de outro locale.

**Importante:** Isso afeta apenas o TREINO (não a predição via API, onde os números vêm como floats do JSON). É um bug pré-existente do projeto original — não foi alterado nesta revisão para não exigir retreinamento.

### 3.4. `test_model.py` depende de `.pkl` no disco

O teste `test_model.py` carrega `app/model/modelo.pkl` com `joblib.load`. Se o modelo não foi treinado localmente, o teste falha. Este é mais um teste de integração do que unitário, e depende do approach `.pkl` que foi abandonado em favor do MLflow.

### 3.5. `testar_api_completo.py` — URL default é 8001

O script E2E usa `BASE_URL_DEFAULT = "http://localhost:8001"` (portas separadas). Se o Henrique usar portas padrão, precisa passar `--url http://localhost:8000`.

---

## 4. Arquivos modificados nesta revisão (para aplicar no Git)

| Arquivo | Tipo de alteração |
|---------|-------------------|
| `app/routes.py` | Corrigido (normalização de texto + Fase numérica + remoção de Pedra_Num) |
| `tests/test_train.py` | Corrigido (mock de os.path.isfile) |
| `tests/test_model.py` | Corrigido (remoção de Pedra_Num, normalização de texto no teste) |
| `.gitignore` | Corrigido (typo logs/ß → logs/) |
| `testar_sistema_completo.py` | Novo (teste abrangente de todo o sistema Docker) |
| `RELATORIO-CORRECOES-REVISAO.md` | Novo (este arquivo) |

---

## 5. Como aplicar as correções

1. Copiar os arquivos modificados para o repositório do Henrique (substituir os existentes).
2. Rodar os testes: `python -m pytest tests/ -v` — todos os 27 testes devem passar.
3. Se o modelo já estiver treinado e em produção, chamar `POST /reload` para recarregar (a normalização de texto passa a funcionar imediatamente).
4. **Não é necessário retreinar o modelo** — as correções alinham a predição com o que o modelo já espera.

---

## 6. Resultado dos testes após correções

```
tests/test_api.py::test_home PASSED
tests/test_api.py::test_predict_risk_high PASSED
tests/test_api.py::test_predict_risk_low PASSED
tests/test_api.py::test_predict_model_not_loaded PASSED
tests/test_api.py::test_prediction_internal_error PASSED
tests/test_api.py::test_model_loading_exception PASSED
tests/test_api.py::test_reload_model_success PASSED
tests/test_api.py::test_reload_model_exception PASSED
tests/test_api.py::test_retrain_endpoint PASSED
tests/test_api.py::test_executar_treinamento_em_background_success PASSED
tests/test_api.py::test_executar_treinamento_em_background_calledprocesserror PASSED
tests/test_api.py::test_executar_treinamento_em_background_general_exception PASSED
tests/test_train.py::test_run_training_pipeline PASSED
tests/test_train.py::test_run_training_file_error PASSED
tests/test_preprocessing.py::test_clean_data_conversion PASSED
tests/test_preprocessing.py::test_clean_data_missing_columns PASSED
tests/test_preprocessing.py::test_clean_data_full_cleaning PASSED
tests/test_feature_engineering.py::test_create_features_success PASSED
tests/test_feature_engineering.py::test_create_features_missing_target PASSED
tests/test_feature_engineering.py::test_extrair_fase_edge_cases PASSED
tests/test_utils.py::test_load_data_success PASSED
tests/test_utils.py::test_load_data_file_not_found PASSED
tests/test_utils.py::test_load_data_missing_defasagem PASSED
tests/test_utils.py::test_load_data_fallback_comma_separator PASSED
tests/test_evaluate.py::test_evaluate_model_metrics PASSED
tests/test_evaluate.py::test_evaluate_model_threshold_influence PASSED
tests/test_evaluate.py::test_evaluate_model_with_pipeline_and_no_importance PASSED

27 passed
```

### Teste de sistema (Docker) — `testar_sistema_completo.py`

```
 1. GET / (Health)                                    → OK
 2. GET /openapi.json                                 → OK (5 paths)
 3. GET /metrics (Prometheus)                         → OK (4 metricas customizadas)
 4. POST /predict (payload valido)                    → OK (risco=1, prob=0.7917)
 5. POST /predict (validacao campos/tipos)            → OK (3/3 retornaram 422)
 6. POST /predict (normalizacao acentos/case)         → OK (10/10 cenarios)
 7. POST /predict (combinacoes Pedra x Fase)          → OK (24/24)
 8. POST /predict (valores extremos)                  → OK (4/4)
 9. POST /reload                                      → OK
10. POST /retrain                                     → OK
11. POST /predict apos /reload                        → OK
12. Prometheus                                        → OK
13. Grafana                                           → OK
14. Loki                                              → AVISO (pode demorar a responder /ready)
15. MLflow UI                                         → OK

29 OK, 1 AVISO, 0 ERROS
```
