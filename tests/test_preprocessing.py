import pandas as pd
import numpy as np
from src.preprocessing import clean_data


def test_clean_data_conversion():
    # Arrange
    df_raw = pd.DataFrame(
        {
            "IAA": ["5,5", "7.0", "8,1"],
            "IEG": [10, "0,0", "NaN"],
            "Defasagem": ["-1", "0", "2"],
        }
    )

    # Act
    df_clean = clean_data(df_raw)

    # Assert
    assert df_clean["IAA"].dtype == float or df_clean["IAA"].dtype == np.float64
    assert df_clean["IEG"].iloc[1] == 0.0
    assert pd.isna(df_clean["IEG"].iloc[2])
    assert df_clean["Defasagem"].iloc[0] == -1.0


def test_clean_data_missing_columns():
    # Arrange
    df_raw = pd.DataFrame({"Outra": [1, 2]})
    # Act
    df_clean = clean_data(df_raw)
    # Assert
    assert "Outra" in df_clean.columns


def test_clean_data_full_cleaning():
    # Arrange: Simula idades erradas, notas acima de 10, e textos com acento/espaço
    df_raw = pd.DataFrame(
        {
            "Idade": [
                15,
                "1/17/00",
                4,
                35,
            ],  # Normal, Data do Excel, Muito baixa, Muito alta
            "IPS": [
                11.0,
                -2.0,
                5.0,
                8.0,
            ],  # Nota acima de 10 deve virar 10, abaixo de 0 vira 0
            "Texto_Cat": [" João ", "Mação", "NaN", "nan"],  # Textos categóricos
            "Defasagem": [1, 2, -1, 0],
        }
    )

    # Act
    df_clean = clean_data(df_raw)

    # Assert - Idades
    assert df_clean["Idade"].iloc[0] == 15.0
    assert df_clean["Idade"].iloc[1] == 17.0  # Converteu a data
    assert pd.isna(df_clean["Idade"].iloc[2])  # Anulou idade 4
    assert pd.isna(df_clean["Idade"].iloc[3])  # Anulou idade 35

    # Assert - Notas (Clipping)
    assert df_clean["IPS"].iloc[0] == 10.0  # Clipou no 10
    assert df_clean["IPS"].iloc[1] == 0.0  # Clipou no 0

    # Assert - Textos
    assert df_clean["Texto_Cat"].iloc[0] == "JOAO"  # Tirou espaço, upper e sem acento
    assert df_clean["Texto_Cat"].iloc[1] == "MACAO"
    assert pd.isna(df_clean["Texto_Cat"].iloc[2])  # "NaN" string virou null real
