import pandas as pd
import numpy as np
import re


def create_features(df):
    """
    Cria o target, aplica engenharia de atributos (interações/hierarquia),
    remove identificadores e colunas de leakage.
    """
    df = df.copy()

    # Criação do Target (Risco) só ocorre se a coluna existir (treinamento)
    if "Defasagem" in df.columns:
        df["alvo_risco"] = np.where(df["Defasagem"] < 0, 1, 0)

    # Criação Nuances de Comportamento
    if all(c in df.columns for c in ["IEG", "IDA", "IAA", "IPS"]):
        df["IEG_x_IDA"] = df["IEG"] * df["IDA"]  # Esforço vs Resultado
        df["IEG_x_IAA"] = df["IEG"] * df["IAA"]  # Esforço vs Autoimagem
        df["IPS_x_IDA"] = df["IPS"] * df["IDA"]  # Psicológico vs Resultado

    # Fase
    if "Fase" in df.columns:
        df["Fase_Num"] = df["Fase"].apply(extrair_fase)

    # Seleção de Features e Remoção de Vazamento (Data Leakage)
    cols_to_drop = [
        "Defasagem",
        "Ano_Base",
        "RA",
        "Nome",
        "Nome Anonimizado",
        "Data de Nasc",
    ]

    # Removemos INDE e IAN (Vazamento de dados confirmados)
    leakage_cols = [
        c for c in df.columns if "INDE" in str(c).upper() or "IAN" in str(c).upper()
    ]
    cols_to_drop.extend(leakage_cols)

    # Executa a remoção
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # Isola o Y
    y = None
    if "alvo_risco" in X.columns:
        y = X["alvo_risco"]
        X = X.drop(columns=["alvo_risco"])

    return X, y


def extrair_fase(fase_str):
    fase_str = str(fase_str).upper()
    if "ALFA" in fase_str or "ALPHA" in fase_str:
        return 0
    match = re.search(r"\d+", fase_str)
    if match:
        return int(match.group())
    return np.nan
