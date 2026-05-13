import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

from preprocessing import preprocess

def get_models() -> dict:
    """
    Devuelve un diccionario con los tres modelos sin entrenar. Los parámetros principales:
        max_iter=1000 — Logistic Regression necesita más iteraciones para converger con este dataset
        n_estimators=100 — Random Forest y XGBoost usarán 100 árboles
        random_state=42 — resultados reproducibles siempre
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }

def train_all(X_train, y_train) -> dict:
    """
    Itera sobre los tres modelos y para cada uno hace dos cosas:
        Primero — validación cruzada:
        divide X_train en 5 partes iguales y repite 5 veces:
    Al final promedia los 5 resultados. Esto da una estimación fiable sin tocar X_test.
        Segundo — entrenamiento final:
        model.fit() entrena con todos los datos de train. Este es el modelo que se guardará y usará en producción.
    """
    models = get_models()
    trained = {}

    for name, model in models.items():
        print(f'Entrenando {name}...')

        # Validación cruzada para estimar rendimiento real
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        print(f'  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

        # Entrenamiento final con todos los datos de train
        model.fit(X_train, y_train)
        trained[name] = model

    return trained

def save_models(trained: dict, path: str = '../models/'):
    """
    Guarda cada modelo entrenado como archivo .pkl (formato binario de Python).
    joblib.dump serializa el objeto — es decir, convierte el modelo entrenado en un archivo que puedes cargar después sin reentrenar.
    """
    os.makedirs(path, exist_ok=True)
    for name, model in trained.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, os.path.join(path, filename))
        print(f'Guardado: {filename}')

def train(data_path: str = '../data/heart.csv'):
    """
    Orquesta todo el flujo:
        Llama a preprocess de preprocessing.py — recibe los datos listos
        Entrena los tres modelos
        Guarda los modelos y el scaler
        Devuelve los modelos y los datos de test para que evaluation.py los evalúe
    El scaler se guarda también porque en producción, cuando llegue un nuevo paciente, 
    necesitarás escalar sus datos con el mismo scaler que usaste en entrenamiento.
    """
    X_train, X_test, y_train, y_test, scaler = preprocess(data_path)

    trained = train_all(X_train, y_train)

    save_models(trained)
    joblib.dump(scaler, '../models/scaler.pkl')

    return trained, X_test, y_test

if __name__ == '__main__':
    train()