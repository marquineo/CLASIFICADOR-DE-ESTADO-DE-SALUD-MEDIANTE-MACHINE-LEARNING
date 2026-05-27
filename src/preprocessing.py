import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """
    Lee el CSV y devuelve un DataFrame. Recibe la ruta como parámetro para que sea reutilizable desde cualquier script.
    """
    df = pd.read_csv(path)
    return df

def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas que identificamos en el EDA a tipo category. Lo mismo que hice en el notebook pero formalizado.
    """
    cols_category = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df[cols_category] = df[cols_category].astype('category')
    df['FastingBS'] = df['FastingBS'].astype('int64')  # ← añadido
    return df

def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dos pasos:
        Primero convierte los 0 a NaN — porque pandas distingue entre "valor cero" y "valor desconocido"
        Luego agrupa por HeartDisease y rellena cada NaN con la mediana de su grupo (enfermos con mediana de enfermos, sanos con mediana de sanos)
    """
    # Reemplazar 0 por NaN para poder imputar
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)

    # Imputar con mediana agrupada por HeartDisease
    for col in ['Cholesterol', 'RestingBP']:
        df[col] = df.groupby('HeartDisease')[col].transform(
            lambda x: x.fillna(x.median())
        )
    return df

def agrupar_categorias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la decisión del EDA: TA tenía solo ~40 casos y distribución 50/50 igual que NAP,
    así que los fusionamos para que el modelo tenga más ejemplos de ese patrón.
    """
    # TA se agrupa con NAP por distribución similar y pocos ejemplos
    df['ChestPainType'] = df['ChestPainType'].replace({'TA': 'NAP'})
    return df

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las categóricas a columnas binarias. Por ejemplo ST_Slope se convierte en tres columnas:
    ST_Slope_Up  ST_Slope_Flat  ST_Slope_Down
     1              0              0
     0              1              0
     0              0              1
    """
    cols_ohe = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    df = pd.get_dummies(df, columns=cols_ohe, drop_first=False)
    # FastingBS ya es binaria (0/1), no necesita encoding
    return df

def scale(df: pd.DataFrame, num_cols: list, scaler=None):
    """
    Escala las columnas numéricas. La lógica del if/else es clave:
        En train — scaler=None → crea el scaler, aprende la media y desviación de train, escala
        En test — recibe el scaler ya entrenado → solo escala sin reaprender
    Esto evita data leakage: si el scaler aprendiera con test, el modelo tendría información del futuro.
    """
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        # En test usamos el scaler entrenado en train
        df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler

def split(df: pd.DataFrame):
    """
    Separa el dataset en:
        X — todas las columnas menos el target
        y — solo HeartDisease
        test_size=0.2 — 80% train, 20% test
        random_state=42 — resultado reproducible siempre
        stratify=y — mantiene la proporción 55/45 en ambos conjuntos
    """
    X = df.drop(columns='HeartDisease')
    y = df['HeartDisease']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def preprocess(path: str):
    """
    Orquesta todas las funciones anteriores en el orden correcto.
    Desde train_model.py solo necesitarás llamar a esta función con la ruta del CSV y recibirás los datos listos para entrenar.
    """
    df = load_data(path)
    df = fix_types(df)
    df = impute_zeros(df)
    df = agrupar_categorias(df)
    df = encode(df)

    num_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    X_train, X_test, y_train, y_test = split(df)

    X_train, scaler = scale(X_train, num_cols)
    X_test, _ = scale(X_test, num_cols, scaler)

    return X_train, X_test, y_train, y_test, scaler