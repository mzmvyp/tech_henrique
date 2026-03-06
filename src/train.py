import joblib
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from utils import load_data
from preprocessing import clean_data
from feature_engineering import create_features
from evaluate import evaluate_model

def run_training():
    print("Iniciando Pipeline de Treinamento com MLFLOW...")
    
    # Cria uma pasta 'mlruns' localmente para salvar os dados
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("PassosMagicos_Risco_Defasagem")

    # Habilita o log automático (salva params, métricas e o modelo .pkl)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False)

    # Definição de Caminhos (usa só os arquivos que existirem)
    paths_base = {
        '2022': 'files/PEDE2022.csv',
        '2023': 'files/PEDE2023.csv',
        '2024': 'files/PEDE2024.csv'
    }
    paths = {ano: p for ano, p in paths_base.items() if os.path.isfile(p)}
    if not paths:
        print("Erro: Nenhum arquivo encontrado em files/ (PEDE2022.csv, PEDE2023.csv ou PEDE2024.csv).")
        return
    print(f"         Arquivos a carregar: {list(paths.keys())}")
    
    # Pipeline de Dados
    print("   [1/6] Carregando dados (Utils)...")
    try:
        df_raw = load_data(paths)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return

    print("   [2/6] Limpando dados (Preprocessing)...")
    df_clean = clean_data(df_raw)
    
    print("   [3/6] Engenharia de Features...")
    X, y = create_features(df_clean)
    
    print(f"         Features finais: {len(X.columns)} colunas identificadas.")

    # Split de Dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Transformador para variáveis numéricas (preenche nulos com a mediana)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # Transformador para variáveis categóricas (texto)
    # Preenche nulos com o valor mais frequente (moda) e converte texto para números (One-Hot)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # O ColumnTransformer aplica as regras corretas usando seletores dinâmicos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, make_column_selector(dtype_exclude="object")),
            ('cat', categorical_transformer, make_column_selector(dtype_include="object"))
        ])

    # Criação do Pipeline Final
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Otimização (Random Search)
    print("   [4/6] Buscando melhores hiperparâmetros...")
    
    # Grade de hiperparâmetros expandida para extrair a máxima performance
    param_dist = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2'], # Importante para diversificar as árvores
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }    
    
    with mlflow.start_run() as run:
        print(f"   [MLflow] Run iniciada. ID: {run.info.run_id}")
        
        # Aumentamos o n_iter para 20 para testar mais combinações e encontrar o melhor modelo
        random_search = RandomizedSearchCV(
            estimator=model_pipeline, 
            param_distributions=param_dist, 
            n_iter=20, 
            cv=3, 
            verbose=1, 
            random_state=42, 
            n_jobs=-1, # Usa todos os núcleos do processador para treinar mais rápido
            scoring='recall' # Otimizando para Recall (encontrar o maior número possível de alunos em risco)
        )
        
        print("   [5/6] Treinando o modelo...")

        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        
        # Avaliação
        print("   [6/6] Avaliando...")        
        evaluate_model(best_model, X_test, y_test, threshold=0.40)

        assinatura = infer_signature(X_test, best_model.predict(X_test))
        
        # Grava o modelo no MLflow com a assinatura e registra oficialmente
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="modelo",
            signature=assinatura,
            registered_model_name="Modelo_Risco_Defasagem"
        )
        
        # Salvar
        output_dir = 'app/model'
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(best_model, 'app/model/modelo.pkl')
        print("\nModelo salvo com sucesso em app/model/modelo.pkl")

if __name__ == "__main__":
    run_training()