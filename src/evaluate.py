import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    recall_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
)
from sklearn.metrics import ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Calcula métricas com base em um limiar de decisão customizado e loga no MLFlow
    """

    print("\n" + "=" * 100)
    print(f"MELHORES PARÂMETROS:")
    print(f"{model.get_params()}")
    print("\n" + "=" * 100)

    # Predição de Probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]

    # Aplicação do Threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Importância das Features e Investigação de Vazamento
    print("\n" + "=" * 100)
    print("IMPORTANCIA DAS FEATURES E INVESTIGAÇÃO DE POSSÍVEL VAZAMENTO DE DADOS:")
    print("=" * 100)

    # Verifica se é um Pipeline e extrai o classificador e os nomes corretos das features geradas
    classifier = model
    feature_names = X_test.columns

    if hasattr(model, "named_steps"):
        if "classifier" in model.named_steps:
            classifier = model.named_steps["classifier"]
        if "preprocessor" in model.named_steps:
            try:
                # Extrai os nomes das colunas após o OneHotEncoding (ex: de 13 para 104)
                feature_names = model.named_steps[
                    "preprocessor"
                ].get_feature_names_out()
            except Exception as e:
                print(
                    f"Aviso: Não foi possível obter nomes das features transformadas: {e}"
                )

    if hasattr(classifier, "feature_importances_"):
        try:
            importances = pd.Series(
                data=classifier.feature_importances_, index=feature_names
            )
            # Mostramos o Top 20 para o terminal não ficar ilegível com 104 colunas
            print(importances.sort_values(ascending=False).head(20))
        except Exception as e:
            print(f"Erro ao mapear importâncias: {e}")
    else:
        print("Feature importance indisponível para este estimador.")

    # Cálculo das Métricas
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        cmap="Blues",
        display_labels=["Sem Risco", "Com Risco"],
    )
    plt.title("Matriz de Confusão")
    mlflow.log_figure(fig, "matriz_confusao.png")
    plt.close(fig)

    print("\n" + "=" * 100)
    print(f"RESULTADOS DA AVALIAÇÃO DO MODELO (Limiar: {threshold})")
    print("=" * 100)
    print(f"ACURÁCIA:              {accuracy:.2%}")
    print(f"PRECISÃO:              {precision:.2%}")
    print(f"* RECALL:              {recall:.2%}")
    print(f"F1-SCORE:              {f1:.2%}")
    print(
        "* RECALL É A MÉTRICA MAIS IMPORANTE POIS ELA REPRESENTA A SENSIBILIDADE DO NOSSO MODELO"
    )
    print("-" * 100)

    print("\nMatriz de Confusão:")
    print(f"Verdadeiros Negativos: {cm[0][0]} | Falsos Positivos: {cm[0][1]}")
    print(f"Falsos Negativos:      {cm[1][0]} | Verdadeiros Positivos: {cm[1][1]}")

    print("\nRelatório Completo:")
    print(classification_report(y_test, y_pred, zero_division=0))

    mlflow.log_param("threshold", threshold)
    mlflow.log_metrics(
        {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    )

    return {"recall": recall, "f1": f1, "accuracy": accuracy, "confusion_matrix": cm}
