import streamlit as st
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicción de Baterías", page_icon="🔋")

# --- FUNCIÓN PARA PROCESAR SMILES ---
def analizar_smiles(smiles):
    """Toma un SMILES y cuenta los átomos para las variables de Kim et al."""
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None, None, None
    
    # Contar átomos específicos en el solvente
    # Ecuación: sum(1 for atom in mol if symbol == 'X')
    sC = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
    sO = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
    sF = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
    
    return sC, sO, sF

# --- CARGAR MODELO ---
try:
    model = joblib.load('modelo_baterias.pkl')
except FileNotFoundError:
    st.error("⚠️ No se encontró el archivo 'modelo_baterias.pkl'. Asegúrate de subirlo a GitHub.")
    st.stop()

# --- INTERFAZ DE USUARIO ---
st.title("🔋 Predictor de Desempeño de Electrolitos")
st.markdown("""
Esta herramienta utiliza **Machine Learning** para predecir si un electrolito tendrá **Alta Conductividad**.
Ingresa la estructura química (SMILES) y las condiciones experimentales.
""")

# --- BLOQUE 1: DATOS QUÍMICOS (AUTOMÁTICO) ---
st.header("1. Composición Química")
smiles_input = st.text_input("Ingresa el SMILES del Solvente:", value="C1COC(=O)O1", help="Ejemplo: Carbonato de Propileno")

# Calculamos las variables automáticamente
sC_calc, sO_calc, sF_calc = analizar_smiles(smiles_input)

if sC_calc is not None:
    st.success(f"✅ Estructura Válida detectada. Átomos contados:")
    col1, col2, col3 = st.columns(3)
    col1.metric("Carbonos (sC)", sC_calc)
    col2.metric("Oxígenos (sO)", sO_calc)
    col3.metric("Flúor (sF)", sF_calc)
else:
    st.error("❌ SMILES inválido. Por favor verifica la cadena.")
    sC_calc, sO_calc, sF_calc = 0, 0, 0 # Valores default para no romper el código

# --- BLOQUE 2: CONDICIONES EXPERIMENTALES (MANUAL) ---
st.header("2. Condiciones Experimentales")
col_a, col_b = st.columns(2)

with col_a:
    temperature = st.number_input("Temperatura (°C)", value=25.0)
    conc_salt = st.number_input("Concentración de Sal (mol/L)", value=1.0)

with col_b:
    # Aquí podrías agregar inputs para la Sal (Anión) si tu modelo los pide
    # Por ahora dejamos valores fijos o inputs manuales si los necesitas
    # Ejemplo: aO (Oxígenos del Anión)
    aO = st.number_input("Oxígenos en el Anión (aO)", value=4) 

# --- PREDICCIÓN ---
if st.button("🔮 Calcular Desempeño", type="primary"):
    
    # IMPORTANTE: El orden de estas columnas debe ser EXACTAMENTE 
    # el mismo con el que entrenaste tu modelo XGBoost/RandomForest.
    # Ajusta esta lista según tu X_train.columns
    
    datos_entrada = pd.DataFrame([[
        temperature, 
        conc_salt, 
        sO_calc,   # Variable calculada por RDKit
        sC_calc,   # Variable calculada por RDKit
        sF_calc,   # Variable calculada por RDKit
        aO         # Variable manual
    ]], columns=['temperature', 'conc_salt', 'sO', 'sC', 'sF', 'aO'])
    
    # Nota: Si tu modelo usa más variables (como FO, FC, etc.), 
    # debes calcularlas aquí antes de crear el DataFrame.
    # Ejemplo: datos_entrada['FO'] = datos_entrada['sF'] / datos_entrada['sO']

    try:
        prediction = model.predict(datos_entrada)[0]
        
        st.divider()
        if prediction == 1:
            st.balloons()
            st.success("### ✅ Resultado: ALTO DESEMPEÑO (> 4 mS/cm)")
            st.info("Este electrolito es apto para carga rápida.")
        else:
            st.error("### ⚠️ Resultado: BAJO DESEMPEÑO (< 4 mS/cm)")
            st.warning("Este electrolito generará alta resistencia interna.")
            
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        st.write("Revisa que el número de columnas en 'datos_entrada' coincida con tu modelo.")
