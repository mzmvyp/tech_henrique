import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.utils import load_data

CSV_CONTENT = "IAA;IEG;IPS;IDA;IPV\n5.5;6.0;7.0;8.0;9.0"


def test_load_data_success():
    mock_df = pd.DataFrame({"IAA": [5.5], "IEG": [6.0], "Defasagem": [-1.0]})
    paths = {"2022": "caminho/falso/2022.csv"}

    with patch("pandas.read_csv", return_value=mock_df) as mock_read:
        df_result = load_data(paths)
        assert isinstance(df_result, pd.DataFrame)
        assert "Ano_Base" in df_result.columns
        assert df_result["Ano_Base"].iloc[0] == 2022
        assert "Defasagem" in df_result.columns


def test_load_data_file_not_found():
    paths = {"2022": "arquivo_inexistente.csv"}
    with patch("pandas.read_csv", side_effect=FileNotFoundError("Arquivo não achado")):
        with pytest.raises(FileNotFoundError):
            load_data(paths)


def test_load_data_missing_defasagem():
    # Arrange: Mock sem a coluna Defasagem para ativar a trava de segurança
    mock_df = pd.DataFrame({"IAA": [5.5], "IEG": [6.0]})
    paths = {"2022": "dummy.csv"}

    # Act & Assert
    with patch("pandas.read_csv", return_value=mock_df):
        with pytest.raises(ValueError, match="A coluna 'Defasagem' não foi encontrada"):
            load_data(paths)


def test_load_data_fallback_comma_separator():
    # Arrange: Força o primeiro read_csv (ponto e vírgula) a falhar para testar o fallback (vírgula)
    mock_df = pd.DataFrame({"IAA": [5.5], "IEG": [6.0], "Defasagem": [-1.0]})
    paths = {"2022": "dummy.csv"}

    # Simula erro na 1ª chamada, retorna dataframe na 2ª
    with patch(
        "pandas.read_csv", side_effect=[Exception("Erro de separador"), mock_df]
    ):
        df_result = load_data(paths)
        assert "Defasagem" in df_result.columns
