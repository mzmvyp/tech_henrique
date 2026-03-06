import joblib
import pandas as pd
import numpy as np
import os
import pytest
from sklearn.pipeline import Pipeline

MODEL_PATH = "app/model/modelo.pkl"


def test_model_file_exists():
    """Verifica se o arquivo .pkl foi gerado no caminho esperado"""
    assert os.path.exists(
        MODEL_PATH
    ), f"O arquivo do modelo não foi encontrado em: {MODEL_PATH}"


def test_model_loading():
    """Verifica se conseguimos carregar o arquivo com joblib"""
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Falha ao carregar o modelo .pkl: {e}")

    assert isinstance(model, Pipeline) or hasattr(
        model, "predict"
    ), "O objeto carregado não parece ser um modelo válido."


def test_model_prediction_mechanics():
    """
    Verifica se o modelo consegue receber um DataFrame com Nulos e devolver uma previsão sem quebrar
    """
    model = joblib.load(MODEL_PATH)

    # Colunas EXATAS que o modelo treinado espera, baseadas no src/train.py e AlunoRequest
    input_data = pd.DataFrame(
        {
            "IAA": [5.5, np.nan],
            "IEG": [6.0, 4.0],
            "IPS": [7.0, 5.0],
            "IDA": [8.0, 6.0],
            "IPV": [9.0, 7.0],
            "Idade": [15, 16],
            "Fase": ["8", "9"],
            "Pedra": ["AGATA", "QUARTZO"],
            "Instituicao_de_ensino": ["ESCOLA PUBLICA", "ONG"],
            "Genero": ["F", "M"],
            "IEG_x_IDA": [48.0, 24.0],
            "IEG_x_IAA": [33.0, np.nan],
            "IPS_x_IDA": [56.0, 30.0],
            "Fase_Num": [8, 9],
        }
    )

    try:
        preds = model.predict(input_data)
        probs = model.predict_proba(input_data)

    except Exception as e:
        pytest.fail(f"O modelo falhou ao realizar a predição: {e}")

    # Verificações de Saída
    assert len(preds) == 2
    assert preds[0] in [0, 1]

    assert probs.shape == (2, 2)
    assert 0.0 <= probs[0][1] <= 1.0
