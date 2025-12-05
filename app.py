import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Calculadora de Electrolitos", page_icon="♦︎")

# --- FUNCIÓN CIENTÍFICA PRINCIPAL ---
def analizar_componente(smiles):
    """Cuenta átomos C, O, F, Li, H en una molécula dada por SMILES"""
    if not smiles or smiles.strip() == "":
        return {'C':0, 'O':0, 'F':0, 'Li':0, 'Total':0}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol) # Importante para contar todo bien
        return {
            'C': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'C']),
            'O': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'O']),
            'F': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'F']),
            'Li': len([a for a in mol.GetAtoms() if a.GetSymbol() == 'Li']),
            'Total': mol.GetNumAtoms()
        }
    except:
        return {'C':0, 'O':0, 'F':0, 'Li':0, 'Total':0}

def calcular_kim_variables(solventes, sales):
    """
    Calcula las variables de Kim et al. para una mezcla arbitraria.
    solventes: Lista de diccionarios [{'smiles', 'ratio', 'densidad', 'mw'}, ...]
    sales: Lista de diccionarios [{'smiles', 'conc'}, ...]
    """
    
    # 1. Calcular MOLES totales de cada átomo en la mezcla (Base 1 Litro)
    moles_O_total = 0
    moles_C_total = 0
    moles_F_total = 0
    atomos_totales_global = 0 # Suma de todos los átomos (Denominador)
    
    # Variables acumuladoras para numerador (Solvente vs Sal)
    O_from_solv = 0
    C_from_solv = 0
    F_from_solv = 0
    O_from_salt = 0
    C_from_salt = 0
    F_from_salt = 0

    # --- PROCESAR SOLVENTES ---
    for solv in solventes:
        if solv['ratio'] > 0:
            # Paso crítico: Volumen -> Masa -> Moles
            # Moles = (Ratio_v/v * 1000 mL * Densidad) / PesoMolecular
            moles_molecula = (solv['ratio'] * 1000 * solv['densidad']) / solv['mw']
            
            atomos = analizar_componente(solv['smiles'])
            
            # Aportación de este solvente a la mezcla
            moles_O_total += moles_molecula * atomos['O']
            moles_C_total += moles_molecula * atomos['C']
            moles_F_total += moles_molecula * atomos['F']
            
            # Acumuladores específicos
            O_from_solv += moles_molecula * atomos['O']
            C_from_solv += moles_molecula * atomos['C']
            F_from_solv += moles_molecula * atomos['F']
            
            atomos_totales_global += moles_molecula * atomos['Total']

    # --- PROCESAR SALES ---
    for sal in sales:
        if sal['conc'] > 0:
            # Moles = Concentración Molar (ya está en moles/L)
            moles_molecula = sal['conc']
            
            atomos = analizar_componente(sal['smiles'])
            
            moles_O_total += moles_molecula * atomos['O']
            moles_C_total += moles_molecula * atomos['C']
            moles_F_total += moles_molecula * atomos['F']
            
            # Acumuladores específicos
            O_from_salt += moles_molecula * atomos['O']
            C_from_salt += moles_molecula * atomos['C']
            F_from_salt += moles_molecula * atomos['F']
            
            atomos_totales_global += moles_molecula * atomos['Total']

    # --- CÁLCULO DE VARIABLES FINALES ---
    if atomos_totales_global == 0:
        return None # Evitar división por cero si no ingresan nada

    # Variables de Fracción Atómica (sO, aO, etc.)
    res = {
        'sO': O_from_solv / atomos_totales_global,
        'sC': C_from_solv / atomos_totales_global,
        'sF': F_from_solv / atomos_totales_global,
        'aO': O_from_salt / atomos_totales_global,
        'aC': C_from_salt / atomos_totales_global,
        'aF': F_from_salt / atomos_totales_global,
    }
    
    # Ratios Químicos
    epsilon = 1e-9
    res['Total_O'] = res['sO'] + res['aO']
    res['Total_C'] = res['sC'] + res['aC']
    res['Total_F'] = res['sF'] + res['aF']
    
    res['FO'] = res['Total_F'] / (res['Total_O'] + epsilon)
    res['FC'] = res['Total_F'] / (res['Total_C'] + epsilon)
    res['OC'] = res['Total_O'] / (res['Total_C'] + epsilon)
    
    return res

# --- INTERFAZ DE USUARIO ---
st.title("Prediccion de Eficiencia Coulombica y Condutividad Ionica en Electrolitos de Baterias de Iones de Litio")
st.markdown("Ingresa tu receta (volumétrica) para calcular automáticamente las 13 variables de Kim et al.")

with st.form("receta_form"):
    st.subheader("1. Solventes (Hasta 4)")
    
    solventes_input = []
    cols = st.columns(4)
    
    # Generamos 4 columnas dinámicas
    for i in range(4):
        with cols[i]:
            st.markdown(f"**Solvente {i+1}**")
            s_smiles = st.text_input(f"SMILES S{i+1}", placeholder="C1CCOC1")
            s_ratio = st.number_input(f"Ratio (0-1) S{i+1}", min_value=0.0, max_value=1.0, step=0.1)
            s_dens = st.number_input(f"Densidad (g/mL) S{i+1}", value=1.0, step=0.01)
            s_mw = st.number_input(f"Peso Mol. (g/mol) S{i+1}", value=100.0, step=1.0)
            
            solventes_input.append({
                'smiles': s_smiles, 'ratio': s_ratio, 'densidad': s_dens, 'mw': s_mw
            })

    st.divider()
    st.subheader("2. Sales (Hasta 2)")
    sales_input = []
    cols_s = st.columns(2)
    
    for i in range(2):
        with cols_s[i]:
            st.markdown(f"**Sal {i+1}**")
            sa_smiles = st.text_input(f"SMILES Sal {i+1}", placeholder="[Li+].F[P-](F)(F)(F)(F)F")
            sa_conc = st.number_input(f"Concentración (M) Sal {i+1}", min_value=0.0, step=0.1)
            
            sales_input.append({'smiles': sa_smiles, 'conc': sa_conc})
            
    calcular_btn = st.form_submit_button("Calcular Variables")

# --- MOSTRAR RESULTADOS ---
if calcular_btn:
    try:
        resultados = calcular_kim_variables(solventes_input, sales_input)
        
        if resultados:
            st.success("¡Cálculo Exitoso!")
            
            # Mostrar las 13 variables bonitas
            st.subheader("Resultados (Variables de Kim et al.)")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("sO (O Solv)", f"{resultados['sO']:.4f}")
            c2.metric("sC (C Solv)", f"{resultados['sC']:.4f}")
            c3.metric("sF (F Solv)", f"{resultados['sF']:.4f}")
            c4.metric("FO Ratio", f"{resultados['FO']:.4f}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("aO (O Sal)", f"{resultados['aO']:.4f}")
            c2.metric("aC (C Sal)", f"{resultados['aC']:.4f}")
            c3.metric("aF (F Sal)", f"{resultados['aF']:.4f}")
            c4.metric("FC Ratio", f"{resultados['FC']:.4f}")
            
            # Dataframe para copiar
            st.caption("Datos listos para tu modelo:")
            df_res = pd.DataFrame([resultados])
            st.dataframe(df_res)
            
        else:
            st.warning("Por favor ingresa al menos 1 solvente y 1 sal válidos.")
            
    except Exception as e:
        st.error(f"Ocurrió un error en el cálculo: {e}")
