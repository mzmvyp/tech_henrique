import sys
import os
import pytest
from unittest.mock import patch

# Obtém o caminho absoluto para a pasta src
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))

if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture(autouse=True)
def bloquear_mlflow_globalmente():
    """
    Fixture de segurança: Garante que o MLflow nunca executa de verdade
    durante os testes unitários, evitando a poluição da base de dados.
    """
    with patch("src.train.mlflow"), patch("src.evaluate.mlflow"), patch(
        "app.routes.mlflow"
    ):
        yield
