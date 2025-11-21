import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Cargar el modelo
# Asegúrate de que el archivo .pkl se llame EXACTAMENTE igual a como lo subiste
model = joblib.load('modelo_baterias.pkl')

# 2. Título e Instrucciones
st.title("🔋 Predicción de Desempeño de Electrolitos")
st.write("Ingresa las propiedades del electrolito para predecir si tendrá Alta Conductividad.")

# 3. Crear los inputs para el usuario
# IMPORTANTE: Estos nombres deben coincidir con el orden en que entrenaste el modelo
# Ajusta los valores por defecto según tus datos reales
temperature = st.number_input("Temperatura (°C)", value=25.0)
conc_salt = st.number_input("Concentración de Sal (mol/L)", value=1.0)
so_val = st.number_input("Oxígeno del Solvente (sO)", value=0.5)
# Agrega aquí el resto de variables si tu modelo usa más...

# 4. Botón de Predicción
if st.button("Predecir Desempeño"):
    # Crear un dataframe con los valores
    input_data = pd.DataFrame([[temperature, conc_salt, so_val]], 
                              columns=['temperature', 'conc_salt', 'sO']) 
    
    try:
        prediction = model.predict(input_data)[0]
        
        # 5. Mostrar resultado
        if prediction == 1:
            st.success("✅ Resultado: ALTO DESEMPEÑO (Apto para carga rápida)")
        else:
            st.error("❌ Resultado: BAJO DESEMPEÑO")
    except Exception as e:
        st.error(f"Error al predecir: {e}")
