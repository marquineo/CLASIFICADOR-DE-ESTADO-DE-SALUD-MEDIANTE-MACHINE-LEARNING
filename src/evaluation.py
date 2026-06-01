import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
import os

from preprocessing import preprocess

def load_models(path: str = '../models/') -> dict:
    """
    Lee la carpeta models/ y carga todos los .pkl excepto el scaler. Convierte el nombre del archivo al nombre legible:
    """
    models = {}
    for filename in os.listdir(path):
        if filename.endswith('.pkl') and filename != 'scaler.pkl':
            name = filename.replace('.pkl', '').replace('_', ' ').title()
            models[name] = joblib.load(os.path.join(path, filename))
    return models

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Dos tipos de predicción:
        y_pred — predicción binaria (0 o 1) — usada para Accuracy, Precision, Recall, F1
        y_prob — probabilidad de ser enfermo (0.0 a 1.0) — usada para AUC
    [:, 1] selecciona la probabilidad de la clase positiva (enfermo), no la negativa.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall':    recall_score(y_test, y_pred),
        'F1':        f1_score(y_test, y_pred),
        'AUC':       roc_auc_score(y_test, y_prob)
    }

def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Itera sobre los tres modelos, calcula sus métricas y las organiza en un DataFrame para comparar fácilmente:
                        Accuracy  Precision  Recall    F1     AUC
    Logistic Regression  0.856     0.871     0.872   0.871   0.923
    Random Forest        0.878     0.891     0.891   0.891   0.941
    XGBoost              0.883     0.897     0.896   0.896   0.948
    """
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test)
        print(f'\n{name}')
        print(classification_report(y_test, model.predict(X_test)))

    return pd.DataFrame(results).T

def plot_confusion_matrices(models: dict, X_test, y_test):
    """
    Muestra una matriz por modelo. Cada celda significa:
                    Predicho Sano   Predicho Enfermo
    Real Sano            TN                FP
    Real Enfermo         FN                TP
    TN — sano predicho como sano
    TP — enfermo predicho como enfermo
    FP — sano predicho como enfermo
    FN — enfermo predicho como sano
    """
    fig, axes = plt.subplots(1, len(models), figsize=(16, 5))

    for i, (name, model) in enumerate(models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Sano', 'Enfermo'],
                    yticklabels=['Sano', 'Enfermo'])
        axes[i].set_title(name, fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Predicho')
        axes[i].set_ylabel('Real')

    plt.suptitle('Matrices de Confusión', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../images/confusion_matrices.png', bbox_inches='tight')
    plt.show()

def plot_roc_curves(models: dict, X_test, y_test):
    """
    Dibuja la curva ROC de los tres modelos en la misma gráfica para comparar. 
    La línea diagonal punteada representa un modelo que adivina al azar — cuanto más arriba y a la izquierda esté la curva, mejor el modelo. 
    El AUC es el área bajo esa curva.
    """
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Azar (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/roc_curves.png', bbox_inches='tight')
    plt.show()

def plot_metrics_comparison(results: pd.DataFrame):
    """
    Gráfico de barras agrupadas con todas las métricas de los tres modelos. Permite ver de un vistazo qué modelo gana en cada métrica.
    """
    results.plot(kind='bar', figsize=(12, 6), colormap='Set2', edgecolor='black')
    plt.title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('../images/metrics_comparison.png', bbox_inches='tight')
    plt.show()

def evaluate_by_sex(models: dict, X_test, y_test, df_original):
    """
    Aplica la conclusión del EDA — vigilar el rendimiento segmentado por sexo. 
    Recupera la columna Sex del CSV original usando el índice de X_test para alinear correctamente, 
    luego calcula el Recall por separado para hombres y mujeres. 
    Se espera ver un Recall más bajo en mujeres por el desbalance detectado.
    """
    # Recuperar la columna Sex del índice original
    sex_test = df_original.loc[X_test.index, 'Sex']

    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f'\n{name} — Recall por sexo:')
        for sex in ['M', 'F']:
            mask = sex_test == sex
            recall = recall_score(y_test[mask], y_pred[mask])
            print(f'  {sex}: {recall:.3f}')

def evaluate(data_path: str = '../data/heart.csv'):
    """
    Orquesta todo el flujo de evaluación en orden:
        Preprocesa los datos para obtener X_test e y_test
        Carga los modelos guardados por train_model.py
        Calcula métricas, genera las tres gráficas y analiza por sexo
    """
    X_train, X_test, y_train, y_test, scaler = preprocess(data_path)
    models = load_models()

    # Métricas generales
    results = evaluate_all(models, X_test, y_test)
    print('\n=== RESUMEN ===')
    print(results.round(3))

    # Gráficas
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_metrics_comparison(results)

    # Vigilar rendimiento por sexo (conclusión del EDA)
    df_original = pd.read_csv(data_path)
    evaluate_by_sex(models, X_test, y_test, df_original)

    return results

if __name__ == '__main__':
    evaluate()