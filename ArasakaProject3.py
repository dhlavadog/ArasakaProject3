# ============================================================
#  PIPELINE COMPLETO PARA CLASIFICACIÓN
#  EDA → PREPARACIÓN → ENTRENAMIENTO → EVALUACIÓN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ============================================================
# 1. CARGA DE DATOS
# ============================================================

print("Cargando datos...")

df = pd.read_csv("Healthcare.csv")
print("Datos cargados correctamente!")
print(df.head())

# ============================================================
# 2. PROCESAMIENTO DE SÍNTOMAS
# ============================================================

print("\nProcesando síntomas...")

df['Symptoms_list'] = df['Symptoms'].apply(lambda x: [s.strip() for s in x.split(",")])
all_symptoms = sorted({symptom for row in df['Symptoms_list'] for symptom in row})

print("Número total de síntomas únicos:", len(all_symptoms))

for symptom in all_symptoms:
    df[f"sym_{symptom}"] = df['Symptoms_list'].apply(lambda x: 1 if symptom in x else 0)

# ============================================================
# 3. EDA BÁSICO
# ============================================================

print("\n=== EDA BÁSICO ===")
print("\nDimensiones del dataset:", df.shape)

print("\nDescripción estadística:")
print(df.describe())

print("\nConteo de clases:")
print(df['Disease'].value_counts())

plt.figure(figsize=(10,5))
df['Disease'].value_counts().plot(kind='bar')
plt.title("Distribución de enfermedades")
plt.show()

# ============================================================
# 4. PREPARACIÓN DEL DATASET (X,y)
# ============================================================

print("\nPreparando dataset para entrenamiento...")

# Eliminamos columnas no útiles
df = df.drop(["Patient_ID", "Symptoms", "Symptoms_list"], axis=1)

# X = características (todos los síntomas + edad + género + Symptom_Count)
X = df.drop("Disease", axis=1)

# Convertir género a numérico
X['Gender'] = X['Gender'].astype('category').cat.codes

# y = objetivo
y = df['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Escalado — para mejorar SVM / NB
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dataset listo para entrenamiento!")
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

# ============================================================
# 5. FUNCIÓN PARA GRAFICAR MATRIZ DE CONFUSIÓN
# ============================================================

def plot_conf_matrix(cm, title, cmap="Greens"):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap=cmap)
    plt.title(f"Matriz de confusión - {title}")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

# ============================================================
# 6. MODELO 1: RANDOM FOREST
# ============================================================

print("\nEntrenando Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
print("Accuracy Random Forest:", rf_acc)

cm_rf = confusion_matrix(y_test, rf_pred)
plot_conf_matrix(cm_rf, "Random Forest", cmap="Greens")

# ============================================================
# 7. MODELO 2: SVM
# ============================================================

print("\nEntrenando SVM...")

svm = SVC(kernel='rbf', C=3, gamma='scale')
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
print("Accuracy SVM:", svm_acc)

cm_svm = confusion_matrix(y_test, svm_pred)
plot_conf_matrix(cm_svm, "SVM", cmap="Oranges")

# ============================================================
# 8. MODELO 3: NAIVE BAYES
# ============================================================

print("\nEntrenando Naive Bayes...")

nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_pred = nb.predict(X_test_scaled)

nb_acc = accuracy_score(y_test, nb_pred)
print("Accuracy Naive Bayes:", nb_acc)

cm_nb = confusion_matrix(y_test, nb_pred)
plot_conf_matrix(cm_nb, "Naive Bayes", cmap="Blues")

# ============================================================
# 9. RESULTADOS FINALES
# ============================================================

print("\n=== RESULTADOS ===")
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print(f"SVM Accuracy:           {svm_acc:.4f}")
print(f"Naive Bayes Accuracy:   {nb_acc:.4f}")

print("\nPipeline completado exitosamente!\n")
