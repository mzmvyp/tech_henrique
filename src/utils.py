import pandas as pd


def load_data(file_paths):
    """
    Carrega e unifica os dados dos anos 2022, 2023 e 2024,
    garantindo a padronização dos nomes das colunas.
    """
    dfs = []

    # Mapas de renomeação para padronizar colunas
    rename_maps = {
        "2022": {
            "INDE 22": "INDE",
            "Pedra 22": "Pedra",
            "Matem": "Mat",
            "Portug": "Por",
            "Inglês": "Ing",
            "Defas": "Defasagem",
            "IAA 2022": "IAA",
            "IEG 2022": "IEG",
            "IPS 2022": "IPS",
            "IDA 2022": "IDA",
            "IPV 2022": "IPV",
            "IAN 2022": "IAN",
        },
        "2023": {
            "INDE 2023": "INDE",
            "Pedra 2023": "Pedra",
            "IAA 2023": "IAA",
            "IEG 2023": "IEG",
            "IPS 2023": "IPS",
            "IDA 2023": "IDA",
            "IPV 2023": "IPV",
            "IAN 2023": "IAN",
            "Defasagem 2023": "Defasagem",
        },
        "2024": {
            "INDE 2024": "INDE",
            "Pedra 2024": "Pedra",
            "IAA 2024": "IAA",
            "IEG 2024": "IEG",
            "IPS 2024": "IPS",
            "IDA 2024": "IDA",
            "IPV 2024": "IPV",
            "IAN 2024": "IAN",
            "Defasagem 2024": "Defasagem",
        },
    }

    for ano, path in file_paths.items():
        try:
            df = pd.read_csv(path, sep=";", encoding="utf-8")
        except:
            df = pd.read_csv(path, sep=",", encoding="utf-8")

        # CORREÇÃO CRÍTICA: Remove espaços invisíveis dos nomes das colunas antes de mapear
        df.columns = df.columns.str.strip()

        if ano in rename_maps:
            cols_to_rename = {
                k: v for k, v in rename_maps[ano].items() if k in df.columns
            }
            df = df.rename(columns=cols_to_rename)

        df["Ano_Base"] = int(ano)
        dfs.append(df)

    target_cols = [
        "INDE",
        "IAA",
        "IEG",
        "IPS",
        "IDA",
        "IPV",
        "IAN",
        "Defasagem",
        "Ano_Base",
        "Idade",
        "Fase",
        "Pedra",
        "Instituicao_de_ensino",
        "Genero",
    ]

    df_full = pd.concat(dfs, ignore_index=True)
    cols_existentes = [c for c in target_cols if c in df_full.columns]

    # Trava de segurança para garantir que a defasagem foi mapeada
    if "Defasagem" not in cols_existentes:
        raise ValueError(
            f"A coluna 'Defasagem' não foi encontrada. Colunas lidas: {df_full.columns.tolist()}"
        )

    return df_full[cols_existentes]
