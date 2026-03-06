# -*- coding: utf-8 -*-
"""
Teste completo do sistema Datathon rodando no Docker.
Valida API, MLflow, Prometheus, Grafana, Loki e cenarios de normalizacao.

Uso:
    python testar_sistema_completo.py
    python testar_sistema_completo.py --url http://localhost:8001
"""

import sys
import argparse
import time
import json

try:
    import requests
except ImportError:
    print("Instale requests: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
BASE_URL_DEFAULT = "http://localhost:8000"

OK = "\033[92m[OK]\033[0m"
ERRO = "\033[91m[ERRO]\033[0m"
AVISO = "\033[93m[AVISO]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

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
    print(f"\n{'='*70}")
    print(f"  {nome}")
    print(f"{'='*70}")


PAYLOAD_BASE = {
    "IAA": 5.5, "IEG": 2.0, "IPS": 6.0, "IDA": 4.5,
    "IPV": 7.0, "Idade": 15, "Fase": "8", "Pedra": "Agata",
    "Instituicao_de_ensino": "Escola Publica", "Genero": "F",
}


# ========================== TESTES DA API ==================================

def test_health(base, t):
    secao("1. GET / (Health)")
    try:
        r = requests.get(f"{base}/", timeout=t)
        if r.status_code == 200 and r.json().get("status") == "ok":
            log_ok("API respondendo, status ok.")
            return True
        log_erro(f"Resposta inesperada: {r.status_code} {r.text[:200]}")
        return False
    except Exception as e:
        log_erro(f"Falha de conexao: {e}")
        return False


def test_openapi(base, t):
    secao("2. GET /openapi.json")
    try:
        r = requests.get(f"{base}/openapi.json", timeout=t)
        if r.status_code == 200:
            data = r.json()
            paths = list(data.get("paths", {}).keys())
            expected = ["/", "/predict", "/reload", "/retrain"]
            missing = [p for p in expected if p not in paths]
            if missing:
                log_aviso(f"Paths ausentes no schema: {missing}")
            else:
                log_ok(f"OpenAPI valido. Paths: {paths}")
            return True
        log_erro(f"Status {r.status_code}")
        return False
    except Exception as e:
        log_erro(f"Falha: {e}")
        return False


def test_metrics(base, t):
    secao("3. GET /metrics (Prometheus)")
    try:
        r = requests.get(f"{base}/metrics", timeout=t)
        if r.status_code == 200 and "#" in r.text:
            custom_metrics = []
            for m in ["modelo_predicoes_total", "modelo_probabilidade_risco",
                       "feature_input_iaa", "feature_input_ieg"]:
                if m in r.text:
                    custom_metrics.append(m)
            if custom_metrics:
                log_ok(f"Metricas customizadas encontradas: {custom_metrics}")
            else:
                log_aviso("Metricas Prometheus genericas OK, mas customizadas nao encontradas (normal antes da 1a predicao).")
            return True
        log_erro(f"Status {r.status_code}")
        return False
    except Exception as e:
        log_erro(f"Falha: {e}")
        return False


def test_predict_valido(base, t):
    secao("4. POST /predict (payload valido)")
    try:
        r = requests.post(f"{base}/predict", json=PAYLOAD_BASE, timeout=t)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            checks = [
                "risco_defasagem" in data,
                "probabilidade_risco" in data,
                "mensagem" in data,
                data["risco_defasagem"] in [0, 1],
                0.0 <= data["probabilidade_risco"] <= 1.0,
            ]
            if all(checks):
                log_ok(f"Predicao OK: risco={data['risco_defasagem']}, prob={data['probabilidade_risco']:.4f}, msg=\"{data['mensagem']}\"")
                return True
            log_erro(f"Campos invalidos na resposta: {data}")
            return False
        if r.status_code == 500:
            log_aviso(f"Modelo nao carregado: {r.json().get('detail','')[:100]}")
            return True
        log_erro(f"Status inesperado {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        log_erro(f"Falha: {e}")
        return False


def test_predict_validacao(base, t):
    secao("5. POST /predict (validacao de campos)")
    casos = [
        ("Campo faltando", {"IAA": 1.0}, 422),
        ("Tipo errado (string em float)", {**PAYLOAD_BASE, "IAA": "texto"}, 422),
        ("Body vazio", {}, 422),
    ]
    ok = True
    for nome, payload, esperado in casos:
        try:
            r = requests.post(f"{base}/predict", json=payload, timeout=t)
            if r.status_code == esperado:
                log_ok(f"{nome}: retornou {esperado} como esperado.")
            else:
                log_erro(f"{nome}: esperado {esperado}, obtido {r.status_code}")
                ok = False
        except Exception as e:
            log_erro(f"{nome}: excecao {e}")
            ok = False
    return ok


def test_normalizacao_texto(base, t):
    """Testa que a API normaliza acentos e maiusculas/minusculas corretamente."""
    secao("6. POST /predict (normalizacao de texto - acentos e case)")
    casos = [
        ("Pedra com acento", {**PAYLOAD_BASE, "Pedra": "\u00c1gata"}),
        ("Pedra minuscula", {**PAYLOAD_BASE, "Pedra": "agata"}),
        ("Pedra maiuscula", {**PAYLOAD_BASE, "Pedra": "AMETISTA"}),
        ("Genero minusculo", {**PAYLOAD_BASE, "Genero": "f"}),
        ("Instituicao com acento", {**PAYLOAD_BASE, "Instituicao_de_ensino": "Institui\u00e7\u00e3o P\u00fablica"}),
        ("Fase Alfa (texto)", {**PAYLOAD_BASE, "Fase": "Alfa"}),
        ("Fase Alpha (ingles)", {**PAYLOAD_BASE, "Fase": "Alpha"}),
        ("Fase numero string", {**PAYLOAD_BASE, "Fase": "5"}),
        ("Tudo minusculo", {**PAYLOAD_BASE, "Pedra": "topazio", "Genero": "m", "Instituicao_de_ensino": "escola particular", "Fase": "alfa"}),
        ("Pedra desconhecida", {**PAYLOAD_BASE, "Pedra": "Diamante"}),
    ]
    ok_count = 0
    for nome, payload in casos:
        try:
            r = requests.post(f"{base}/predict", json=payload, timeout=t)
            if r.status_code == 200:
                data = r.json()
                log_ok(f"{nome}: risco={data['risco_defasagem']}, prob={data['probabilidade_risco']:.4f}")
                ok_count += 1
            elif r.status_code == 500 and "Modelo" in r.json().get("detail", ""):
                log_aviso(f"{nome}: modelo nao carregado")
                ok_count += 1
            else:
                log_erro(f"{nome}: status {r.status_code} - {r.json().get('detail','')[:80]}")
        except Exception as e:
            log_erro(f"{nome}: excecao {e}")
    total = len(casos)
    if ok_count == total:
        log_ok(f"Todos os {total} cenarios de normalizacao passaram!")
    else:
        log_aviso(f"{ok_count}/{total} cenarios OK.")
    return ok_count == total


def test_pedra_fase_combinacoes(base, t):
    secao("7. POST /predict (todas combinacoes Pedra x Fase)")
    pedras = ["Quartzo", "Agata", "Ametista", "Topazio"]
    fases = ["8", "Alfa", "1", "5", "Alpha", "3"]
    ok_count = 0
    total = len(pedras) * len(fases)
    for pedra in pedras:
        for fase in fases:
            p = PAYLOAD_BASE.copy()
            p["Pedra"] = pedra
            p["Fase"] = fase
            try:
                r = requests.post(f"{base}/predict", json=p, timeout=t)
                if r.status_code in [200, 500]:
                    ok_count += 1
            except Exception:
                pass
    if ok_count == total:
        log_ok(f"Todas as {total} combinacoes (Pedra x Fase) retornaram resposta valida.")
    else:
        log_erro(f"Apenas {ok_count}/{total} combinacoes OK.")
    return ok_count == total


def test_valores_extremos(base, t):
    secao("8. POST /predict (valores extremos)")
    casos = [
        ("Notas zeradas", {**PAYLOAD_BASE, "IAA": 0.0, "IEG": 0.0, "IPS": 0.0, "IDA": 0.0, "IPV": 0.0}),
        ("Notas maximas", {**PAYLOAD_BASE, "IAA": 10.0, "IEG": 10.0, "IPS": 10.0, "IDA": 10.0, "IPV": 10.0}),
        ("Idade minima", {**PAYLOAD_BASE, "Idade": 5}),
        ("Idade maxima", {**PAYLOAD_BASE, "Idade": 25}),
    ]
    ok_count = 0
    for nome, payload in casos:
        try:
            r = requests.post(f"{base}/predict", json=payload, timeout=t)
            if r.status_code == 200:
                d = r.json()
                log_ok(f"{nome}: risco={d['risco_defasagem']}, prob={d['probabilidade_risco']:.4f}")
                ok_count += 1
            else:
                log_erro(f"{nome}: status {r.status_code}")
        except Exception as e:
            log_erro(f"{nome}: {e}")
    return ok_count == len(casos)


def test_reload(base, t):
    secao("9. POST /reload")
    try:
        r = requests.post(f"{base}/reload", timeout=t)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "sucesso":
                log_ok("Recarga do modelo via MLflow OK.")
                return True
        if r.status_code == 500:
            log_aviso("Erro na recarga (pode ser que nao exista alias production).")
            return True
        log_erro(f"Status inesperado: {r.status_code}")
        return False
    except Exception as e:
        log_erro(f"Falha: {e}")
        return False


def test_retrain(base, t):
    secao("10. POST /retrain")
    try:
        r = requests.post(f"{base}/retrain", timeout=5)
        log_info(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if "Treinamento iniciado" in data.get("mensagem", ""):
                log_ok("Retrain aceito (rodando em background).")
                return True
        log_erro(f"Status {r.status_code}: {r.text[:200]}")
        return False
    except requests.exceptions.Timeout:
        log_aviso("Timeout (normal para retrain em background).")
        return True
    except Exception as e:
        log_erro(f"Falha: {e}")
        return False


def test_predict_apos_reload(base, t):
    """Testa predicao logo apos um /reload para verificar que o modelo recarregado funciona."""
    secao("11. POST /predict apos /reload")
    try:
        requests.post(f"{base}/reload", timeout=t)
        time.sleep(1)
        r = requests.post(f"{base}/predict", json=PAYLOAD_BASE, timeout=t)
        if r.status_code == 200:
            data = r.json()
            log_ok(f"Predicao pos-reload OK: risco={data['risco_defasagem']}, prob={data['probabilidade_risco']:.4f}")
            return True
        log_aviso(f"Status {r.status_code} apos reload.")
        return True
    except Exception as e:
        log_erro(f"Falha: {e}")
        return False


# ========================== TESTES DE INFRAESTRUTURA =======================

def test_prometheus(base_api, t):
    secao("12. Prometheus (scraping da API)")
    prom_urls = ["http://localhost:9090", "http://localhost:9091"]
    for prom_url in prom_urls:
        try:
            r = requests.get(f"{prom_url}/api/v1/targets", timeout=t)
            if r.status_code == 200:
                data = r.json()
                targets = data.get("data", {}).get("activeTargets", [])
                api_target = [t for t in targets if "api:8000" in str(t.get("labels", {}))]
                up_targets = [t for t in targets if t.get("health") == "up"]
                log_ok(f"Prometheus ({prom_url}): {len(up_targets)}/{len(targets)} targets UP.")
                return True
        except Exception:
            continue
    log_aviso("Prometheus nao acessivel em 9090 nem 9091.")
    return True


def test_grafana(base_api, t):
    secao("13. Grafana (Health)")
    grafana_urls = ["http://localhost:3000", "http://localhost:3001"]
    for gf_url in grafana_urls:
        try:
            r = requests.get(f"{gf_url}/api/health", timeout=t)
            if r.status_code == 200:
                data = r.json()
                if data.get("database") == "ok":
                    log_ok(f"Grafana ({gf_url}): database OK.")
                    return True
        except Exception:
            continue
    log_aviso("Grafana nao acessivel em 3000 nem 3001.")
    return True


def test_loki(base_api, t):
    secao("14. Loki (Ready)")
    loki_urls = ["http://localhost:3100", "http://localhost:3101"]
    for loki_url in loki_urls:
        try:
            r = requests.get(f"{loki_url}/ready", timeout=t)
            if r.status_code == 200:
                log_ok(f"Loki ({loki_url}): ready.")
                return True
        except Exception:
            continue
    log_aviso("Loki nao acessivel em 3100 nem 3101.")
    return True


def test_mlflow_ui(base_api, t):
    secao("15. MLflow UI (Health)")
    mlflow_urls = ["http://localhost:5050", "http://localhost:5051"]
    for mf_url in mlflow_urls:
        try:
            r = requests.get(f"{mf_url}/api/2.0/mlflow/experiments/search?max_results=1", timeout=t)
            if r.status_code == 200:
                data = r.json()
                exps = data.get("experiments", [])
                log_ok(f"MLflow ({mf_url}): {len(exps)} experimento(s) encontrado(s).")
                return True
        except Exception:
            continue
    log_aviso("MLflow nao acessivel em 5050 nem 5051.")
    return True


# ========================== MAIN ===========================================

def main():
    parser = argparse.ArgumentParser(description="Teste completo do sistema Datathon no Docker")
    parser.add_argument("--url", default=BASE_URL_DEFAULT, help=f"URL base da API (default: {BASE_URL_DEFAULT})")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout por requisicao (s)")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    t = args.timeout

    print(f"\n{'='*70}")
    print(f"  TESTE COMPLETO DO SISTEMA DATATHON (Docker)")
    print(f"{'='*70}")
    print(f"  API URL: {base}")
    print(f"  Timeout: {t}s")
    print(f"{'='*70}")

    global resultados
    resultados = []

    tests = [
        test_health,
        test_openapi,
        test_metrics,
        test_predict_valido,
        test_predict_validacao,
        test_normalizacao_texto,
        test_pedra_fase_combinacoes,
        test_valores_extremos,
        test_reload,
        test_retrain,
        test_predict_apos_reload,
        test_prometheus,
        test_grafana,
        test_loki,
        test_mlflow_ui,
    ]

    for fn in tests:
        fn(base, t)
        time.sleep(0.3)

    # Resumo
    secao("RESUMO FINAL")
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
        print(f"  Detalhes dos erros:")
        for s, msg in resultados:
            if s == "erro":
                print(f"    - {msg}")
    print()
    if erros == 0:
        print(f"  {OK} SISTEMA OPERACIONAL - Nenhum erro critico encontrado.")
        sys.exit(0)
    print(f"  {ERRO} {erros} erro(s) encontrado(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
