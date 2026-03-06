import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import importlib
import sys
import subprocess

from app.main import app
from app import routes
from app.routes import executar_treinamento_em_background

client = TestClient(app)


# Fixture para Mockar o Modelo
@pytest.fixture
def mock_model():
    model_mock = MagicMock()
    # Simula retorno: 20% classe 0 (sem risco), 80% classe 1 (risco)
    model_mock.predict_proba.return_value = [[0.2, 0.8]]
    return model_mock


def test_home():
    # Arrange & Act & Assert
    response = client.get("/")
    assert response.status_code == 200
    assert response.json().get("status") == "ok"


def test_predict_risk_high(mock_model):
    # Arrange & Act & Assert
    with patch.object(routes, "model", mock_model):
        payload = {
            "IAA": 5.5,
            "IEG": 2.0,
            "IPS": 6.0,
            "IDA": 4.5,
            "IPV": 7.0,
            "Idade": 15,
            "Fase": "8",
            "Pedra": "AGATA",
            "Instituicao_de_ensino": "Escola Publica",
            "Genero": "F",
        }
        response = client.post("/predict", json=payload)

        # Se falhar aqui, você pode usar print(response.json()) para ver o motivo
        assert response.status_code == 200
        data = response.json()
        assert data["risco_defasagem"] == 1
        assert "ALERTA" in data["mensagem"]


def test_predict_risk_low():
    # Arrange & Act & Assert
    low_risk_model = MagicMock()
    low_risk_model.predict_proba.return_value = [[0.9, 0.1]]

    with patch.object(routes, "model", low_risk_model):
        payload = {
            "IAA": 10.0,
            "IEG": 10.0,
            "IPS": 10.0,
            "IDA": 10.0,
            "IPV": 10.0,
            "Idade": 15,
            "Fase": "8",
            "Pedra": "AGATA",
            "Instituicao_de_ensino": "Escola Publica",
            "Genero": "F",
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json()["risco_defasagem"] == 0
        assert "baixo" in response.json()["mensagem"]


def test_predict_model_not_loaded():
    # Arrange & Act & Assert
    with patch.object(routes, "model", None):
        payload = {
            "IAA": 0.0,
            "IEG": 0.0,
            "IPS": 0.0,
            "IDA": 0.0,
            "IPV": 0.0,
            "Idade": 15,
            "Fase": "8",
            "Pedra": "AGATA",
            "Instituicao_de_ensino": "Escola Publica",
            "Genero": "F",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Modelo não carregado" in response.json()["detail"]


def test_prediction_internal_error():
    # Arrange & Act & Assert
    mock_model_error = MagicMock()
    mock_model_error.predict_proba.side_effect = Exception("Erro interno matemático")

    with patch.object(routes, "model", mock_model_error):
        payload = {
            "IAA": 5.5,
            "IEG": 6.0,
            "IPS": 7.0,
            "IDA": 8.0,
            "IPV": 9.0,
            "Idade": 15,
            "Fase": "8",
            "Pedra": "AGATA",
            "Instituicao_de_ensino": "Escola Publica",
            "Genero": "F",
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 500
        assert "Erro na predição" in response.json()["detail"]


def test_model_loading_exception():
    # Arrange & Act & Assert
    with patch("mlflow.sklearn.load_model", side_effect=Exception("Erro MLflow")):
        with patch.dict("sys.modules", {"prometheus_client": MagicMock()}):
            importlib.reload(routes)
            assert routes.model is None

    # LIMPEZA SEGURA (TEARDOWN)
    with patch("mlflow.sklearn.load_model", return_value="Modelo Recuperado"):
        with patch.dict("sys.modules", {"prometheus_client": MagicMock()}):
            importlib.reload(routes)

    assert routes.model == "Modelo Recuperado"


def test_reload_model_success():
    # Arrange & Act
    with patch(
        "app.routes.mlflow.sklearn.load_model", return_value="Novo Modelo Atualizado"
    ):
        response = client.post("/reload")

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == "sucesso"
    assert "Modelo atualizado" in response.json()["mensagem"]
    assert routes.model == "Novo Modelo Atualizado"


def test_reload_model_exception():
    # Arrange & Act
    with patch(
        "app.routes.mlflow.sklearn.load_model",
        side_effect=Exception("Conexão com MLflow perdida"),
    ):
        response = client.post("/reload")

    # Assert
    assert response.status_code == 500
    assert "Erro ao recarregar o modelo" in response.json()["detail"]


def test_retrain_endpoint():
    # O TestClient do FastAPI gerencia BackgroundTasks automaticamente
    response = client.post("/retrain")

    assert response.status_code == 200
    assert response.json()["status"] == "sucesso"
    assert "Treinamento iniciado" in response.json()["mensagem"]


@patch("app.routes.subprocess.run")
def test_executar_treinamento_em_background_success(mock_subprocess_run):
    mock_result = MagicMock()
    mock_result.stdout = "Iniciando...\nModelo salvo com sucesso no MLflow!"
    mock_subprocess_run.return_value = mock_result

    executar_treinamento_em_background()

    mock_subprocess_run.assert_called_once()
    args, kwargs = mock_subprocess_run.call_args
    assert args[0] == ["python", "src/train.py"]
    assert kwargs["capture_output"] is True


@patch("app.routes.subprocess.run")
def test_executar_treinamento_em_background_calledprocesserror(mock_subprocess_run):
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["python", "src/train.py"], stderr="Traceback: Memory Error"
    )

    executar_treinamento_em_background()
    mock_subprocess_run.assert_called_once()


@patch("app.routes.subprocess.run")
def test_executar_treinamento_em_background_general_exception(mock_subprocess_run):
    mock_subprocess_run.side_effect = Exception("Permissão negada")

    executar_treinamento_em_background()
    mock_subprocess_run.assert_called_once()
