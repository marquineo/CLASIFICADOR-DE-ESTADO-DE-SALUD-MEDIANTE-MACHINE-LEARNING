# Proyecto 3 — Predicción de Enfermedad Cardíaca mediante Machine Learning

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

El dataset seleccionado contiene información clínica relacionada con enfermedades cardíacas.

## Variables principales

| Variable | Descripción |
|---|---|
| Age | Edad del paciente |
| Sex | Sexo del paciente |
| ChestPainType | Tipo de dolor torácico |
| RestingBP | Presión arterial en reposo |
| Cholesterol | Nivel de colesterol |
| FastingBS | Glucosa en ayunas |
| RestingECG | Resultado del electrocardiograma |
| MaxHR | Frecuencia cardíaca máxima |
| ExerciseAngina | Angina inducida por ejercicio |
| Oldpeak | Alteración del segmento ST |
| ST_Slope | Pendiente del segmento ST |
| HeartDisease | Variable objetivo |

## Variable objetivo

- `0` → No presenta enfermedad cardíaca
- `1` → Presenta enfermedad cardíaca

---

# Tecnologías utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Estructura del proyecto

```bash
CLASIFICADOR-DE-ESTADO-DE-SALUD-MEDIANTE-MACHINE-LEARNING\
│
├── data/
│   └── heart.csv
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