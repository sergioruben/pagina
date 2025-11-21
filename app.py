{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6afaa7fe-ea95-47f0-91dd-2096a3e55bf1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# 1. Cargar el modelo\n",
    "model = joblib.load('modelo_baterias.pkl')\n",
    "\n",
    "# 2. Título e Instrucciones\n",
    "st.title(\"🔋 Predicción de Desempeño de Electrolitos\")\n",
    "st.write(\"Ingresa las propiedades del electrolito para predecir si tendrá Alta Conductividad.\")\n",
    "\n",
    "# 3. Crear los inputs para el usuario (Basado en tus gráficas SHAP)\n",
    "# Ajusta estos nombres a las columnas exactas que usó tu modelo\n",
    "temperature = st.number_input(\"Temperatura (°C)\", value=25.0)\n",
    "conc_salt = st.number_input(\"Concentración de Sal (mol/L)\", value=1.0)\n",
    "so_val = st.number_input(\"Oxígeno del Solvente (sO)\", value=0.5)\n",
    "# ... agrega aquí el resto de tus variables importantes ...\n",
    "\n",
    "# 4. Botón de Predicción\n",
    "if st.button(\"Predecir Desempeño\"):\n",
    "    # Crear un dataframe con los valores (en el mismo orden que tu entrenamiento)\n",
    "    input_data = pd.DataFrame([[temperature, conc_salt, so_val]], \n",
    "                              columns=['temperature', 'conc_salt', 'sO']) \n",
    "                              # Asegúrate de poner TODAS tus columnas aquí\n",
    "    \n",
    "    prediction = model.predict(input_data)[0]\n",
    "    \n",
    "    # 5. Mostrar resultado\n",
    "    if prediction == 1:\n",
    "        st.success(\"✅ Resultado: ALTO DESEMPEÑO (Apto para carga rápida)\")\n",
    "    else:\n",
    "        st.error(\"❌ Resultado: BAJO DESEMPEÑO\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
