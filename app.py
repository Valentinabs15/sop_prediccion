import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar modelo
modelo = joblib.load('modelo_sop.pkl')

# Configuración inicial
st.set_page_config(page_title="Predicción SOP", layout="centered")

# Menú lateral
menu = st.sidebar.radio("Menú:", ["👁 Vista de Datos", "📈 Resultados del Modelo", "🧪 Clasificar Nuevo Dato"])

# Cargar dataset desde GitHub (ajusta esta URL si cambiaste el nombre del archivo)
df = pd.read_csv("https://raw.githubusercontent.com/Valentinabs15/sop_prediccion/main/pcos_dataset.csv")
X = df[['Age', 'BMI', 'Menstrual_Irregularity', 'Testosterone_Level(ng/dL)', 'Antral_Follicle_Count']]
y = df['PCOS_Diagnosis']
y_pred = modelo.predict(X)

# Sección 1: Vista de datos
if menu == "👁 Vista de Datos":
    st.title("Vista Previa del Dataset de SOP")
    st.dataframe(df.head(10))

# Sección 2: Resultados del modelo
elif menu == "📈 Resultados del Modelo":
    st.title("Resultados del Clasificador")
    
    st.subheader("📊 Métricas de Clasificación")
    st.code(classification_report(y, y_pred))

    st.subheader("🔷 Matriz de Confusión")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No SOP", "SOP"], yticklabels=["No SOP", "SOP"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    st.pyplot(fig)

# Sección 3: Clasificador interactivo
elif menu == "🧪 Clasificar Nuevo Dato":
    st.title("Clasificador de SOP (Síndrome de Ovario Poliquístico)")

    edad = st.number_input("Edad", min_value=10, max_value=60, value=25)
    bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
    irregularidad = st.selectbox("¿Tiene irregularidad menstrual?", ["Sí", "No"])
    testosterona = st.number_input("Testosterona (ng/dL)", min_value=0.0, max_value=200.0, value=50.0)
    foliculos = st.number_input("Recuento de folículos antrales", min_value=0, max_value=50, value=20)

    irregular = 1 if irregularidad == "Sí" else 0
    entrada = np.array([[edad, bmi, irregular, testosterona, foliculos]])

    if st.button("Predecir"):
        pred = modelo.predict(entrada)
        if pred[0] == 1:
            st.error("🔴 Diagnóstico Positivo de SOP")
        else:
            st.success("🟢 No hay SOP según el modelo")
