import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from motor_geotecnico import modelar_flujo

st.set_page_config(page_title="Geotecnia Pro: Simulador de Flujo", layout="wide")

st.title("🌊 Análisis de Flujo en Medios Porosos")
st.markdown("---")

# Barra lateral de parámetros (ACTUALIZADA con controles de H1, H2 y Posición)
with st.sidebar:
    st.header("⚙️ Parámetros de Diseño")
    lx = st.slider("Longitud del Dominio (Lx)", 50.0, 180.0, 135.0, 5.0)
    prof_muro = st.select_slider("Profundidad de Tablestaca (m)", options=[0, 5, 10, 15, 20], value=10)
    # RESTAURADO: Slider para la posición del muro (de 0 a 15m, el ancho de la presa)
    pos_muro = st.slider("Posición de Tablestaca (m)", 0.0, 15.0, 5.0, 0.5)
    
    # NUEVO: Controles para las alturas del flujo (H1 y H2)
    st.subheader("💧 Alturas del Flujo")
    h1 = st.number_input("Altura Aguas Arriba (H1)", value=50.0, step=1.0)
    h2 = st.number_input("Altura Aguas Abajo (H2)", value=5.0, step=1.0)
    
    st.subheader("🧪 Propiedades del Suelo")
    k = st.number_input("Permeabilidad (k) [m/s]", value=1e-5, format="%.1e")
    gs = st.number_input("Gravedad Específica (Gs)", value=2.65)
    e = st.number_input("Relación de Vacíos (e)", value=0.65)

# Ejecución del motor con los nuevos parámetros
res = modelar_flujo(lx, prof_muro, pos_muro, dx=0.5, k=k, Gs=gs, e=e, h1=h1, h2=h2)

# Layout de pestañas académicas
tab1, tab2, tab3 = st.tabs(["📊 Resultados y Gráficas", "📖 Marco Teórico", "🛡️ Análisis de Seguridad"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Caudal de Fuga (Q)", f"{res['Q']*1000:.4f} L/s/m")
    col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
    col3.metric("FS Sifonamiento", f"{res['fs']:.2f}")

    # Visualización principal (ESTILO RESTAURADO SEGÚN IMAGEN 2)
    fig, ax = plt.subplots(figsize=(12, 6))
    nx, ny, dx, x_ini, x_fin, y_base = res['coords']
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dx)
    
    # 1. Mapa de Calor (Carga Hidráulica h)
    # Se usa un colormap de gradiente (red/blue/orange) para h
    cf = ax.contourf(X, Y, res['H'], levels=30, cmap='turbo', alpha=0.9)
    # plt.colorbar(cf, ax=ax, label='Carga Hidráulica (h)') # Opcional: barra de colores
    
    # 2. Líneas Equi-potenciales
    # Contornos negros sobre el mapa de calor
    ax.contour(X, Y, res['H'], levels=15, colors='black', linewidths=0.5)
    
    # 3. Líneas de Corriente (masked para no atravesar estructura)
    ix_m = np.ma.masked_array(res['ix'], mask=~res['mask'])
    iy_m = np.ma.masked_array(res['iy'], mask=~res['mask'])
    ax.streamplot(X, Y, ix_m, iy_m, color='navy', linewidth=0.8, density=1.5)
    
    # Dibujo de estructura (gray) como bloque sólido
    ax.imshow(np.where(res['mask'], np.nan, 1), extent=[0, lx, 0, 30], cmap='gray', alpha=0.8, zorder=10)
    
    # Título dinámico
    ax.set_title(f"Red de Flujo y Mapa de Calor (Lx={lx}m, Muro Prof={prof_muro}m a Pos={pos_muro}m, H1={h1}m, H2={h2}m)")
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
