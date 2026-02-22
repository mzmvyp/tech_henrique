import pandas as pd
import numpy as np

def clean_data(df):
    """
    Realiza a limpeza profunda e conversão de tipos dos dados.
    """
    df = df.copy()

    # Limpeza da Variável Alvo Classe
    if 'Defasagem' in df.columns:
        # Tenta converter para número
        df['Defasagem'] = pd.to_numeric(df['Defasagem'], errors='coerce')
        df = df.dropna(subset=['Defasagem'])

    # Tratamento de idade no formato 1/17/00
    if 'Idade' in df.columns:
        def corrigir_idade_excel(valor):
            valor = str(valor).strip()
            if '/' in valor:
                partes = valor.split('/')
                if len(partes) >= 2:
                    return partes[1] # Retorna o 17
            return valor
            
        df['Idade'] = df['Idade'].apply(corrigir_idade_excel)        

    # Tratamento das Colunas Numéricas
    numeric_cols = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV', 'IAN', 'INDE', 'Idade']
    
    for col in numeric_cols:
        if col in df.columns:
            # Converte para string para manipulação
            s = df[col].astype(str)
            # Formato BR usa vírgula como decimal (ex: "5,783" ou "7.055,00")
            # Se contém vírgula, é formato BR: remove ponto de milhar, troca vírgula por ponto
            # Se não contém vírgula, já está em formato internacional ou é inteiro
            mask_br = s.str.contains(',', na=False)
            s_br = s[mask_br].str.replace('.', '', regex=False).str.replace(',', '.')
            s_int = s[~mask_br]  # Formato internacional - mantém como está
            df.loc[mask_br, col] = s_br
            df.loc[~mask_br, col] = s_int
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Transforma Idades incorretas e notas erradas em nulo (NaN)
    if 'Idade' in df.columns:
        df.loc[(df['Idade'] < 5) | (df['Idade'] > 30), 'Idade'] = np.nan    


    notas_cols = ['IAA', 'IEG', 'IPS', 'IDA', 'IPV']
    for col in notas_cols:
        if col in df.columns:
            # Notas nunca podem passar de 10
            df[col] = df[col].clip(lower=0, upper=10) 

    # Limpeza Geral de Colunas de Texto (Categóricas)
    colunas_texto = df.select_dtypes(include=['object']).columns                        
    
    for col in colunas_texto:        
        df[col] = (
            df[col].astype(str)
            .str.normalize('NFKD')
            .str.encode('ascii', errors='ignore')
            .str.decode('utf-8')
            .str.strip()
            .str.upper()
        )
        
        df[col] = df[col].replace('NAN', np.nan)

    return df