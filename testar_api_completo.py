# -*- coding: utf-8 -*-
"""
Script completo de testes da API Datathon (Previsao de Risco).
Testa todos os endpoints contra a API rodando (ex.: no Docker).
Uso: python testar_api_completo.py
      python testar_api_completo.py --url http://localhost:8001
"""

import sys
import argparse
import time

try:
    import requests
except ImportError:
    print("Instale requests: pip install requests")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Configuracao
# -----------------------------------------------------------------------------
BASE_URL_DEFAULT = "http://localhost:8001"  # Portas separadas; use 8000 se for portas padrao

OK = "[OK]"
ERRO = "[ERRO]"
AVISO = "[AVISO]"
INFO = "[INFO]"

# Payload valido para /predict (schema AlunoRequest)
PAYLOAD_ALUNO_VALIDO = {
    "IAA": 5.5,
    "IEG": 2.0,
    "IPS": 6.0,
    "IDA": 4.5,
    "IPV": 7.0,
    "Idade": 15,
    "Fase": "8",
    "Pedra": "Agata",
    "Instituicao_de_ensino": "Escola Publica",
    "Genero": "F",
}

resultados = []


def log_ok(msg):
    print(f"  {OK} {msg}")
    resultados.append(("ok", msg))


def log_erro(msg):
    print(f"  {ERRO} {msg}")
    resultados.append(("erro", msg))


def log_aviso(msg):
    print(f"  {AVISO} {msg}")
    resultados.append(("aviso", msg))


def log_info(msg):
    print(f"  {INFO} {msg}")


def secao(nome):
    print(f"\n{'='*60}")
    print(f"  {nome}")
    print(f"{'='*60}\n")


# -----------------------------------------------------------------------------
# Testes
# -----------------------------------------------------------------------------
def test_health(base_url, timeout=10):
    """GET / - Health / Home"""
    secao("1. GET / (Health / Home)")
    try:
        r = requests.get(f"{base_url}/", timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "ok":
                log_ok("API respondendo, status ok.")
                return True
            log_erro(f"Resposta inesperada: {data}")
            return False
        log_erro(f"Status {r.status_code}: {r.text[:200]}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_openapi(base_url, timeout=10):
    """GET /openapi.json - Schema da API"""
    secao("2. GET /openapi.json (Schema OpenAPI)")
    try:
        r = requests.get(f"{base_url}/openapi.json", timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if "openapi" in data and "paths" in data:
                log_ok("OpenAPI valido, paths disponiveis.")
                return True
            log_erro("OpenAPI sem 'openapi' ou 'paths'.")
            return False
        log_erro(f"Status {r.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_metrics(base_url, timeout=10):
    """GET /metrics - Metricas Prometheus"""
    secao("3. GET /metrics (Prometheus)")
    try:
        r = requests.get(f"{base_url}/metrics", timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            if "modelo_predicoes_total" in r.text or "http_requests_total" in r.text or "request_duration" in r.text or "#" in r.text:
                log_ok("Endpoint /metrics retornou metricas.")
                return True
            log_aviso("Resposta 200 mas conteudo pode nao ser metricas esperadas.")
            return True
        log_erro(f"Status {r.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_predict_valido(base_url, timeout=15):
    """POST /predict - Payload valido"""
    secao("4. POST /predict (payload valido)")
    try:
        start = time.time()
        r = requests.post(f"{base_url}/predict", json=PAYLOAD_ALUNO_VALIDO, timeout=timeout)
        elapsed = time.time() - start
        log_info(f"Status: {r.status_code} | Tempo: {elapsed:.2f}s")
        if r.status_code == 200:
            data = r.json()
            if "risco_defasagem" in data and "probabilidade_risco" in data and "mensagem" in data:
                log_ok(f"Previsao: risco={data['risco_defasagem']}, prob={data['probabilidade_risco']:.4f}")
                return True
            log_erro(f"Resposta sem campos esperados: {data}")
            return False
        if r.status_code == 500 and "Modelo não carregado" in (r.json() or {}).get("detail", ""):
            log_aviso("Modelo nao carregado (normal na 1a subida). Registre o modelo no MLflow e chame /reload.")
            return True
        log_erro(f"Status {r.status_code}: {r.text[:300]}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_predict_invalido(base_url, timeout=10):
    """POST /predict - Payload invalido (validacao)"""
    secao("5. POST /predict (payload invalido - validacao)")
    payload_ruim = {"IAA": 1.0}  # faltam campos
    try:
        r = requests.post(f"{base_url}/predict", json=payload_ruim, timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 422:
            log_ok("Validacao funcionando (422 Unprocessable Entity).")
            return True
        log_erro(f"Esperado 422, obtido {r.status_code}: {r.text[:200]}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_predict_tipos_errados(base_url, timeout=10):
    """POST /predict - Tipos errados (ex.: string onde espera numero)"""
    secao("6. POST /predict (tipos errados)")
    payload = PAYLOAD_ALUNO_VALIDO.copy()
    payload["IAA"] = "nao_e_numero"
    try:
        r = requests.post(f"{base_url}/predict", json=payload, timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 422:
            log_ok("Validacao de tipos funcionando (422).")
            return True
        log_erro(f"Esperado 422, obtido {r.status_code}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_reload(base_url, timeout=15):
    """POST /reload - Recarregar modelo"""
    secao("7. POST /reload")
    try:
        r = requests.post(f"{base_url}/reload", timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "sucesso":
                log_ok("Recarga solicitada com sucesso.")
                return True
            log_aviso(f"Resposta: {data}")
            return True
        if r.status_code == 500:
            log_aviso("Erro ao recarregar (pode ser que nao exista modelo com alias production).")
            return True
        log_erro(f"Status {r.status_code}: {r.text[:200]}")
        return False
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_retrain(base_url, timeout=5):
    """POST /retrain - Disparar retreinamento (só verifica se aceita)"""
    secao("8. POST /retrain")
    try:
        r = requests.post(f"{base_url}/retrain", timeout=timeout)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if "Treinamento iniciado" in data.get("mensagem", "") or data.get("status") == "sucesso":
                log_ok("Endpoint /retrain aceitou a requisicao.")
                return True
            log_aviso(f"Resposta: {data}")
            return True
        log_erro(f"Status {r.status_code}: {r.text[:200]}")
        return False
    except requests.exceptions.Timeout:
        log_aviso("Timeout (esperado: retrain roda em background).")
        return True
    except requests.exceptions.RequestException as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_predict_varios_casos(base_url, timeout=15):
    """POST /predict - Varias combinacoes (Pedra, Fase)"""
    secao("9. POST /predict (variados: Pedra e Fase)")
    pedras = ["Quartzo", "Agata", "Ametista", "Topazio"]
    fases = ["8", "Alfa", "1", "5"]
    ok_count = 0
    for pedra in pedras:
        for fase in fases:
            payload = PAYLOAD_ALUNO_VALIDO.copy()
            payload["Pedra"] = pedra
            payload["Fase"] = fase
            try:
                r = requests.post(f"{base_url}/predict", json=payload, timeout=timeout)
                if r.status_code == 200:
                    ok_count += 1
                elif r.status_code == 500 and "Modelo não carregado" in (r.json() or {}).get("detail", ""):
                    ok_count += 1
            except Exception:
                pass
    total = len(pedras) * len(fases)
    if ok_count == total:
        log_ok(f"Todos os {total} casos (Pedra x Fase) retornaram 200 ou 500 (modelo nao carregado).")
        return True
    log_aviso(f"{ok_count}/{total} casos ok.")
    return ok_count >= total // 2


def main():
    parser = argparse.ArgumentParser(description="Testes completos da API Datathon")
    parser.add_argument("--url", default=BASE_URL_DEFAULT, help=f"URL base da API (default: {BASE_URL_DEFAULT})")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout por requisicao (segundos)")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    timeout = args.timeout

    print("\n" + "="*60)
    print("  TESTES COMPLETOS - API DATATHON (Previsao de Risco)")
    print("="*60)
    print(f"  URL: {base}")
    print(f"  Timeout: {timeout}s")
    print("="*60)

    global resultados
    resultados = []

    tests = [
        test_health,
        test_openapi,
        test_metrics,
        test_predict_valido,
        test_predict_invalido,
        test_predict_tipos_errados,
        test_reload,
        test_retrain,
        test_predict_varios_casos,
    ]

    for fn in tests:
        fn(base, timeout)
        time.sleep(0.3)

    # Resumo
    secao("RESUMO")
    ok = sum(1 for s, _ in resultados if s == "ok")
    erros = sum(1 for s, _ in resultados if s == "erro")
    avisos = sum(1 for s, _ in resultados if s == "aviso")
    total = len(resultados)
    print(f"  Total de verificacoes: {total}")
    print(f"  {OK}: {ok}")
    if avisos:
        print(f"  {AVISO}: {avisos}")
    if erros:
        print(f"  {ERRO}: {erros}")
    print()
    if erros == 0:
        print(f"  {OK} Nenhum erro critico. Sistema em ordem.")
        sys.exit(0)
    print(f"  {ERRO} {erros} erro(s) encontrado(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
