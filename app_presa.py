import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from motor_geotecnico import modelar_flujo

st.set_page_config(page_title="Geotecnia Pro: Simulador de Flujo", layout="wide")

st.title("🌊 Análisis de Flujo en Medios Porosos")
st.markdown("---")

# Barra lateral de parámetros
with st.sidebar:
    st.header("⚙️ Parámetros de Diseño")
    lx = st.slider("Longitud del Dominio (Lx)", 50.0, 180.0, 135.0, 5.0)
    prof_muro = st.select_slider("Profundidad de Tablestaca (m)", options=[0, 5, 10, 15, 20], value=10)
    
    st.subheader("🧪 Propiedades del Suelo")
    k = st.number_input("Permeabilidad (k) [m/s]", value=1e-5, format="%.1e")
    gs = st.number_input("Gravedad Específica (Gs)", value=2.65)
    e = st.number_input("Relación de Vacíos (e)", value=0.65)

# Ejecución del motor
res = modelar_flujo(lx, prof_muro, dx=0.5, k=k, Gs=gs, e=e)

# Layout de pestañas académicas
tab1, tab2, tab3 = st.tabs(["📊 Resultados y Gráficas", "📖 Marco Teórico", "🛡️ Análisis de Seguridad"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Caudal de Fuga (Q)", f"{res['Q']*1000:.4f} L/s/m")
    col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
    col3.metric("FS Sifonamiento", f"{res['fs']:.2f}")

    # Visualización principal
    fig, ax = plt.subplots(figsize=(12, 6))
    nx, ny, dx, x_ini, x_fin, y_base = res['coords']
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dx)
    
    # Red de flujo con enmascaramiento de líneas
    ax.contourf(X, Y, res['H'], levels=30, cmap='Blues_r', alpha=0.3)
    # Enmascarar ix/iy para streamplot
    ix_m = np.ma.masked_array(res['ix'], mask=~res['mask'])
    iy_m = np.ma.masked_array(res['iy'], mask=~res['mask'])
    ax.streamplot(X, Y, ix_m, iy_m, color='navy', linewidth=0.8, density=1.5)
    
    # Dibujo de estructura (Concreto)
    ax.imshow(np.where(res['mask'], np.nan, 1), extent=[0, lx, 0, 30], cmap='gray', alpha=0.8, zorder=10)
    
    ax.set_title(f"Red de Flujo (Dominio Lx={lx}m, Muro={prof_muro}m)")
    st.pyplot(fig)

with tab2:
    st.header("Formulación Matemática")
    st.latex(r"\nabla^2 h = \frac{\partial^2 h}{\partial x^2} + \frac{\partial^2 h}{\partial y^2} = 0")
    st.markdown("**Ley de Darcy (Caudal Unitario):**")
    st.latex(r"q = k \cdot i \cdot A \implies Q = \int_{0}^{H} k \frac{\partial h}{\partial x} dy")
    st.markdown("**Gradiente Crítico de Ebullición:**")
    st.latex(r"i_c = \frac{G_s - 1}{1 + e}")

with tab3:
    st.header("Análisis de Estabilidad")
    if res['fs'] > 1.5:
        st.success(f"Sistema Seguro. FS = {res['fs']:.2f} (> 1.5)")
    elif res['fs'] > 1.0:
        st.warning(f"Condición Crítica. FS = {res['fs']:.2f} (Margen insuficiente)")
    else:
        st.error(f"Falla por Sifonamiento Detectada. FS = {res['fs']:.2f} (<= 1.0)")
