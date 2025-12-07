# 🧠 Clasificación de Enfermedades con Machine Learning

Este proyecto implementa un pipeline completo de Machine Learning para **clasificar enfermedades** usando técnicas de:
- **Random Forest**
- **Support Vector Machines (SVM)**
- **Naive Bayes**

El modelo utiliza el dataset **Healthcare.csv**, el cual incluye datos como edad, género, síntomas y número de síntomas, para predecir la enfermedad correspondiente.

Este dataset puede ser encontrado en *Kaggle* como:
[**Healthcare Symptoms–Disease Classification Dataset**](https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset)

---
## 📊 ¿Qué hace el pipeline?

### 1️⃣ **Carga del dataset**
Lee `Healthcare.csv`, imprime las primeras filas y revisa la estructura.

### 2️⃣ **EDA básico**
- Dimensiones.  
- Estadísticas descriptivas.  
- Conteo de clases.  
- Gráfica de distribución de enfermedades.  

### 3️⃣ **Preprocesamiento**
- Separación X / Y.  
- División en train/test.  
- Escalado estándar (muy importante para SVM).

### 4️⃣ **Entrenamiento de modelos**
Entrena los tres clasificadores y genera predicciones.

### 5️⃣ **Evaluación**
- Accuracy por modelo.  
- Matrices de confusión (plot con seaborn).  
- Comparación de rendimientos.  

### 6️⃣ **Resultados finales**
Imprime un resumen claro.

---

## 📁 Estructura del Proyecto

```bah
ArasakaProject3
├── Healthcare.csv
├── requirements.txt
├── ArasakaProject3.py
└── README.md
```


- **Healthcare.csv** → Dataset principal  
- **ArasakaProject3.py** → Código principal del pipeline (EDA, preparación, entrenamiento y evaluación)  
- **requirements.txt** → Librerías necesarias para ejecutarlo  

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/dhlavadog/ArasakaProject3.git
cd ArasakaProject3
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```
---
## ▶️ Ejecución

```bash
python ArasakaProject.py
```

El script realiza automáticamente:

1. Carga del dataset

2. EDA (análisis exploratorio de datos)

3. Preparación del dataset

4. Entrenamiento de los modelos

5. Evaluación (accuracy + matriz de confusión)

Durante la ejecución se abrirán varias gráficas relacionadas con:

* Distribución de enfermedades
* Matrices de confusión para cada modelo

**Importante:** Estas graficas se ejecutan una a una, para ver la siguiente toca cerrar la que esté abierta en ese momento.

---
## 🧬 Dataset: Healthcare.csv

El dataset tiene la siguiente estructura:

|Columna|	Descripción
|-|-|
|Patient_ID|	ID único del paciente
|Age|	Edad
|Gender|	Género
|Symptoms|	Lista de síntomas
|Symptom_Count|	Número de síntomas
|Disease|	Etiqueta a predecir
---
## 📊 Modelos incluidos

|Modelo|	Descripción|
|-|-|
|Random Forest|	Basado en múltiples árboles de decisión, robusto y estable.
|SVM (RBF)|	Muy bueno para problemas con fronteras complejas.
|Naive Bayes|	Basado en probabilidad; útil como baseline.


