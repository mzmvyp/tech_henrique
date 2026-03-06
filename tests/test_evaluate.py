import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.evaluate import evaluate_model


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.8, 0.2], [0.2, 0.8], [0.54, 0.41]])
    model.feature_importances_ = np.array([0.1, 0.2, 0.7])
    return model


@patch("src.evaluate.mlflow")
@patch("src.evaluate.ConfusionMatrixDisplay.from_estimator")
def test_evaluate_model_metrics(mock_cmd, mock_mlflow, mock_model):
    X_test = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    y_test = pd.Series([0, 1, 1])

    results = evaluate_model(mock_model, X_test, y_test, threshold=0.40)

    assert results["f1"] == 1.0
    assert results["recall"] == 1.0
    cm = results["confusion_matrix"]
    assert cm[0][0] == 1
    assert cm[1][1] == 2
    mock_cmd.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()


@patch("src.evaluate.mlflow")
@patch("src.evaluate.ConfusionMatrixDisplay.from_estimator")
def test_evaluate_model_threshold_influence(mock_cmd, mock_mlflow, mock_model):
    X_test = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
    y_test = pd.Series([0, 1, 1])
    results = evaluate_model(mock_model, X_test, y_test, threshold=0.90)
    assert results["recall"] == 0.0


@patch("src.evaluate.mlflow")
@patch("src.evaluate.ConfusionMatrixDisplay.from_estimator")
def test_evaluate_model_with_pipeline_and_no_importance(mock_cmd, mock_mlflow):
    # Simula um Pipeline real onde o classificador NÃO tem feature_importances (ex: SVM)
    pipeline_mock = MagicMock()
    pipeline_mock.predict_proba.return_value = np.array([[0.8, 0.2], [0.2, 0.8]])

    # Simula a estrutura named_steps
    classifier_mock = MagicMock()
    del classifier_mock.feature_importances_  # Remove a propriedade

    preprocessor_mock = MagicMock()
    preprocessor_mock.get_feature_names_out.return_value = np.array(["feat1", "feat2"])

    pipeline_mock.named_steps = {
        "classifier": classifier_mock,
        "preprocessor": preprocessor_mock,
    }

    X_test = pd.DataFrame({"col1": [1, 2]})
    y_test = pd.Series([0, 1])

    # Act
    results = evaluate_model(pipeline_mock, X_test, y_test)

    # Assert
    assert "f1" in results
