# -*- coding: utf-8 -*-
"""
Converte o Excel 'BASE DE DADOS PEDE 2024 - DATATHON.xlsx' para files/PEDE2024.csv
no formato esperado pelo pipeline de treino (utils.load_data).
Execute na raiz do projeto: python scripts/xlsx_para_pede2024.py
"""

import os
import sys

try:
    import pandas as pd
except ImportError:
    print("Instale pandas: pip install pandas openpyxl")
    sys.exit(1)

# Caminhos
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(RAIZ, "files", "BASE DE DADOS PEDE 2024 - DATATHON.xlsx")
CSV_SAIDA = os.path.join(RAIZ, "files", "PEDE2024.csv")

# Mapeamento: nome no Excel -> nome esperado pelo pipeline (target_cols)
RENAME_2024 = {
    "INDE 22": "INDE",
    "Pedra 22": "Pedra",
    "IAA": "IAA",
    "IEG": "IEG",
    "IPS": "IPS",
    "IDA": "IDA",
    "IPV": "IPV",
    "IAN": "IAN",
    "Defas": "Defasagem",
    "Idade 22": "Idade",
    "Gênero": "Genero",
    "Género": "Genero",
    "Instituição de ensino": "Instituicao_de_ensino",
    "Instituicao de ensino": "Instituicao_de_ensino",
}

COLUNAS_FINAIS = [
    "INDE", "IAA", "IEG", "IPS", "IDA", "IPV", "IAN",
    "Defasagem", "Idade", "Fase", "Pedra", "Instituicao_de_ensino", "Genero",
]


def main():
    if not os.path.isfile(XLSX):
        print(f"Arquivo nao encontrado: {XLSX}")
        sys.exit(1)

    print(f"Lendo: {XLSX}")
    df = pd.read_excel(XLSX)
    df.columns = df.columns.str.strip()

    # Renomear colunas (só as que existem)
    cols_rename = {k: v for k, v in RENAME_2024.items() if k in df.columns}
    df = df.rename(columns=cols_rename)

    # Manter só colunas que existem e que o pipeline usa
    cols_ok = [c for c in COLUNAS_FINAIS if c in df.columns]
    if "Defasagem" not in cols_ok:
        print("AVISO: Coluna Defasagem nao encontrada. Colunas disponiveis:", list(df.columns))
    df_out = df[cols_ok].copy()

    # Remover linhas com Defasagem nula (se existir) para nao quebrar o treino
    if "Defasagem" in df_out.columns:
        df_out = df_out.dropna(subset=["Defasagem"])

    os.makedirs(os.path.dirname(CSV_SAIDA), exist_ok=True)
    df_out.to_csv(CSV_SAIDA, sep=";", index=False, encoding="utf-8")
    print(f"Salvo: {CSV_SAIDA} ({len(df_out)} linhas, {len(cols_ok)} colunas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
