# Proyecto 3 — Clasificador de Enfermedad Cardíaca mediante Machine Learning

## Descripción del proyecto
Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning clásico capaz de predecir la presencia de enfermedad cardíaca a partir de datos biomédicos de pacientes.
El sistema utiliza un dataset clínico en formato CSV y aplica técnicas de análisis de datos, preprocesamiento y clasificación supervisada para detectar posibles casos de riesgo cardiovascular.

---

# Objetivos
- Analizar un dataset médico real.
- Realizar limpieza y preprocesamiento de datos.
- Aplicar técnicas de análisis exploratorio (EDA).
- Entrenar modelos de Machine Learning clásico.
- Evaluar el rendimiento de los modelos predictivos.
- Comparar distintos algoritmos de clasificación.

---

# Dataset utilizado
El dataset contiene 918 registros clínicos de pacientes con 11 variables predictoras y 1 variable objetivo.

## Variables principales
| Variable | Tipo | Descripción |
|---|---|---|
| Age | Numérica | Edad del paciente |
| Sex | Categórica | Sexo del paciente (M/F) |
| ChestPainType | Categórica | Tipo de dolor torácico (ASY/NAP/ATA/TA) |
| RestingBP | Numérica | Presión arterial en reposo |
| Cholesterol | Numérica | Nivel de colesterol |
| FastingBS | Binaria | Glucosa en ayunas > 120 mg/dl (0/1) |
| RestingECG | Categórica | Resultado del electrocardiograma |
| MaxHR | Numérica | Frecuencia cardíaca máxima |
| ExerciseAngina | Categórica | Angina inducida por ejercicio (Y/N) |
| Oldpeak | Numérica | Alteración del segmento ST |
| ST_Slope | Categórica | Pendiente del segmento ST (Up/Flat/Down) |
| HeartDisease | Binaria | Variable objetivo (0/1) |

## Variable objetivo
- `0` → No presenta enfermedad cardíaca (~45%)
- `1` → Presenta enfermedad cardíaca (~55%)

Dataset balanceado — no requiere técnicas de resampling.

---

# Hallazgos del EDA

## Valores problemáticos detectados
- **Cholesterol y RestingBP** contienen valores 0 biológicamente imposibles — tratados como valores perdidos e imputados con la mediana agrupada por HeartDisease.
- **Cholesterol** presenta valores entre 450-600 — outliers reales de hipercolesterolemia severa, mantenidos en el dataset.

## Variables más relevantes para la predicción
- **Oldpeak** — correlación 0.40 con HeartDisease. Valores más altos en enfermos.
- **MaxHR** — correlación -0.40. Enfermos alcanzan menor frecuencia cardíaca máxima.
- **ST_Slope** — Flat fuertemente asociado a enfermos, Up a sanos.
- **ASY (ChestPainType)** — gran mayoría de casos son enfermos.
- **Sex** — hombres con 62% de enfermos vs mujeres con apenas 3%.

## Limitaciones detectadas
- **Desbalance por sexo** — 78% hombres / 22% mujeres. Posible sesgo histórico de recopilación. El modelo es menos fiable en mujeres.
- **TA (ChestPainType)** — solo 40 casos con distribución 50/50, agrupado con NAP en preprocesamiento.

---

# Preprocesamiento

| Paso | Detalle |
|---|---|
| Imputación | Ceros de Cholesterol y RestingBP reemplazados por mediana agrupada por HeartDisease |
| Agrupación | ChestPainType: TA fusionado con NAP |
| Encoding | One-Hot Encoding en variables categóricas |
| Escalado | StandardScaler en variables numéricas |
| Split | 80% train / 20% test con stratify |

---

# Modelos entrenados

Se entrenaron y compararon tres modelos con validación cruzada de 5 folds:

| Modelo | CV Accuracy |
|---|---|
| Logistic Regression | 84.9% ± 2.6% |
| Random Forest | 84.6% ± 1.9% |
| XGBoost | 84.9% ± 1.0% |

---

# Resultados

## Métricas sobre el conjunto de test

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 89.7% | 89.5% | 92.2% | 90.8% | 0.935 |
| **Random Forest** | **90.8%** | 89.7% | **94.1%** | **91.9%** | 0.933 |
| XGBoost | 88.6% | 90.1% | 89.2% | 89.7% | 0.944 |

## Rendimiento por sexo (Recall)

| Modelo | Hombres | Mujeres |
|---|---|---|
| Logistic Regression | 93.8% | 66.7% |
| **Random Forest** | **94.8%** | **83.3%** |
| XGBoost | 90.6% | 66.7% |

---

# Modelo seleccionado: Random Forest

Random Forest fue seleccionado como modelo final por los siguientes motivos:
- **Recall más alto (94.1%)** — minimiza enfermos no detectados, prioritario en contexto médico.
- **Mejor rendimiento en mujeres (83.3%)** — más equitativo que los demás modelos.
- **Accuracy y F1 más altos** de forma global.

### Limitación principal
El modelo rinde significativamente peor en mujeres (83.3%) que en hombres (94.8%) debido al desbalance detectado en el EDA. Los resultados en pacientes femeninas deben interpretarse con cautela.

---

# Tecnologías utilizadas
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

# Estructura del proyecto
```bash
CLASIFICADOR-DE-ESTADO-DE-SALUD-MEDIANTE-MACHINE-LEARNING\
│
├── data/
│   └── heart.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── analisis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluation.py
│
├── images/
│
├── README.md
└── requirements.txt
```

---

# Cómo ejecutar el proyecto

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar los modelos
cd src
python train_model.py

# 3. Evaluar los modelos
python evaluation.py
```
