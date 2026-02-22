import pandas as pd
import numpy as np
import re

def create_features(df):
    """
    Cria o target, aplica engenharia de atributos (interações/hierarquia),
    remove identificadores e colunas de leakage.
    """
    df = df.copy()
    
    # Criação do Target (Risco)
    if 'Defasagem' not in df.columns:
        raise ValueError("Coluna 'Defasagem' necessária para criar o target.")
        
    # Defasagem Negativa = Risco (Classe 1)
    df['alvo_risco'] = np.where(df['Defasagem'] < 0, 1, 0)
    
    # Criação Nuances de Comportamento
    if all(c in df.columns for c in ['IEG', 'IDA', 'IAA', 'IPS']):
        df['IEG_x_IDA'] = df['IEG'] * df['IDA']  # Esforço vs Resultado
        df['IEG_x_IAA'] = df['IEG'] * df['IAA']  # Esforço vs Autoimagem
        df['IPS_x_IDA'] = df['IPS'] * df['IDA']  # Psicológico vs Resultado    
    
    # Engenharia de Variáveis Categóricas
    # As pedras representam uma evolução
    # Após clean_data(), os textos ficam em UPPERCASE e sem acentos (NFKD → ASCII)
    pedra_map = {
        'QUARTZO': 1, 'Quartzo': 1,
        'AGATA': 2, 'Ágata': 2, 'Agata': 2,
        'AMETISTA': 3, 'Ametista': 3,
        'TOPAZIO': 4, 'Topázio': 4, 'Topazio': 4,
    }
    
    # Procura todas as colunas que tenham Pedra no nome e cria uma versão numérica
    colunas_pedra = [c for c in df.columns if 'Pedra' in str(c)]
    for p_col in colunas_pedra:
        nova_col = f"{p_col}_Num"
        df[nova_col] = df[p_col].map(pedra_map)

    # Corrige a coluna Fase
    if 'Fase' in df.columns:
        df['Fase_Num'] = df['Fase'].apply(extrair_fase)        

    # Seleção de Features e Remoção de Vazamento (Data Leakage)
    # Removemos colunas originais de Pedra/Fase (texto) pois já temos as versões numéricas
    cols_to_drop = ['Defasagem', 'Ano_Base', 'RA', 'Nome', 'Nome Anonimizado', 'Data de Nasc']
    cols_to_drop.extend([c for c in df.columns if 'Pedra' in str(c) and '_Num' not in str(c)])
    if 'Fase' in df.columns:
        cols_to_drop.append('Fase')

    # Removemos INDE e IAN (Vazamento de dados confirmados)    
    leakage_cols = [c for c in df.columns if 'INDE' in str(c).upper() or 'IAN' in str(c).upper()]
    cols_to_drop.extend(leakage_cols)
    
    # Executa a remoção
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # Garante que o target não fique nas variáveis explicativas (X)
    if 'alvo_risco' in X.columns:
        X = X.drop(columns=['alvo_risco'])
        
    y = df['alvo_risco']

    return X, y

def extrair_fase(fase_str):
    fase_str = str(fase_str).upper()
    # Trata o caso especial do ALFA (que não tem número)
    if 'ALFA' in fase_str or 'ALPHA' in fase_str:
        return 0
    
    # Busca o primeiro número dentro do texto (ex: "8E" -> 8, "FASE 1" -> 1)
    match = re.search(r'\d+', fase_str)
    if match:
        return int(match.group())
        
    return np.nan # Se vier um texto bizarro sem número, vira nulo