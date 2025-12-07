# ============================================================
#  PIPELINE COMPLETO PARA CLASIFICACIÓN
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# ============================================================
# 1. CARGA DE DATOS
# ============================================================

df = pd.read_csv("Healthcare.csv")
print("Datos cargados correctamente!")
print(df.head())

# ============================================================
# 2. PROCESAMIENTO DE SÍNTOMAS COMO TEXTO (TF-IDF)
# ============================================================

print("\nProcesando síntomas con TF-IDF...")

# TF-IDF aplicado a texto completo de síntomas
vectorizer = TfidfVectorizer()
X_symptoms = vectorizer.fit_transform(df["Symptoms"])

print("Dimensión del vector TF-IDF:", X_symptoms.shape)

# ============================================================
# 3. EDA BÁSICO
# ============================================================

print("\n=== EDA BÁSICO ===")
print("Dimensiones del dataset:", df.shape)

print("\nConteo de enfermedades:")
print(df["Disease"].value_counts())

plt.figure(figsize=(12,5))
df['Disease'].value_counts().plot(kind='bar')
plt.title("Distribución de enfermedades")
plt.xticks(rotation=90)
plt.show()

# ============================================================
# 4. PREPARACIÓN DEL DATASET
# ============================================================

print("\nPreparando dataset...")

# y = objetivo
y = df["Disease"]

# Train-test split del texto
X_train, X_test, y_train, y_test = train_test_split(
    X_symptoms, y, test_size=0.2, stratify=y, random_state=42
)

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

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Accuracy Random Forest:", rf_acc)

cm_rf = confusion_matrix(y_test, rf_pred)
plot_conf_matrix(cm_rf, "Random Forest")

# ============================================================
# 7. MODELO 2: SVM LINEAL (ideal para TF-IDF)
# ============================================================

print("\nEntrenando SVM...")

svm = LinearSVC()
svm.fit(X_train, y_train)

svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("Accuracy SVM:", svm_acc)

cm_svm = confusion_matrix(y_test, svm_pred)
plot_conf_matrix(cm_svm, "SVM", cmap="Oranges")

# ============================================================
# 8. MODELO 3: NAIVE BAYES MULTINOMIAL (ideal para texto)
# ============================================================

print("\nEntrenando Naive Bayes...")

nb = MultinomialNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
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

print("\nPipeline completado exitosamente!")
