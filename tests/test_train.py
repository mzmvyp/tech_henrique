import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.train import run_training

@patch('src.train.load_data')
@patch('src.train.clean_data')
@patch('src.train.create_features')
@patch('src.train.train_test_split')
@patch('src.train.RandomizedSearchCV') 
@patch('src.train.evaluate_model')
@patch('src.train.joblib.dump')
@patch('src.train.mlflow')
@patch('src.train.infer_signature') 
@patch('os.makedirs')
@patch('os.path.isfile', return_value=True)
def test_run_training_pipeline(mock_isfile, mock_makedirs, mock_infer, mock_mlflow, mock_dump, mock_eval, mock_search, 
                               mock_split, mock_features, mock_clean, mock_load):
    
    # 1. Configurando os retornos dos Mocks para o fluxo seguir
    mock_load.return_value = pd.DataFrame({'raw': [1]})
    mock_clean.return_value = pd.DataFrame({'clean': [1]})
    mock_features.return_value = (pd.DataFrame({'X': [1]}), pd.Series([1])) 
    
    mock_split.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # 2. Mock do objeto RandomizedSearchCV e seu .fit
    mock_search_instance = mock_search.return_value
    
    # 3. Criamos um mock real para o modelo para suportar o .predict() da assinatura
    mock_best_estimator = MagicMock()
    mock_best_estimator.predict.return_value = [0] 
    mock_search_instance.best_estimator_ = mock_best_estimator
    
    # Act
    run_training()
    
    # Assert - Verifica se cada etapa foi chamada corretamente
    mock_load.assert_called_once()
    mock_clean.assert_called_once()
    mock_features.assert_called_once()
    mock_split.assert_called_once()
    mock_search_instance.fit.assert_called_once()
    mock_eval.assert_called_once()
    
    # Garante que o MLflow inferiu a assinatura e registrou o modelo
    mock_infer.assert_called_once()
    mock_mlflow.sklearn.log_model.assert_called_once()
    

@patch('src.train.load_data')
@patch('src.train.mlflow')
@patch('os.path.isfile', return_value=True)
def test_run_training_file_error(mock_isfile, mock_mlflow, mock_load):
    # Arrange
    mock_load.side_effect = FileNotFoundError("Arquivo sumiu")
    
    # Act
    run_training()
    
    # Assert
    mock_load.assert_called_once()