import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from motor_geotecnico import modelar_flujo

st.set_page_config(page_title="Geotecnia Pro: Simulador de Flujo", layout="wide")

st.title("🌊 Análisis de Flujo en Medios Porosos")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Parámetros de Diseño")
    lx = st.slider("Longitud del Dominio (Lx)", 50.0, 180.0, 135.0, 5.0)
    prof_muro = st.slider("Profundidad de Tablestaca (m)", 0.0, 20.0, 10.0, 1.0)
    pos_muro = st.slider("Posición de Tablestaca (m)", 0.0, 15.0, 5.0, 0.5)
    
    st.subheader("💧 Alturas del Flujo")
    h1 = st.number_input("Altura Aguas Arriba (H1)", value=50.0, step=1.0)
    h2 = st.number_input("Altura Aguas Abajo (H2)", value=5.0, step=1.0)
    
    st.subheader("🧪 Propiedades del Suelo")
    k = st.number_input("Permeabilidad (k) [m/s]", value=1e-5, format="%.1e")
    gs = st.number_input("Gravedad Específica (Gs)", value=2.65)
    e = st.number_input("Relación de Vacíos (e)", value=0.65)

# Ejecución
res = modelar_flujo(lx, prof_muro, pos_muro, dx=0.5, k=k, Gs=gs, e=e, h1=h1, h2=h2)

tab1, tab2, tab3 = st.tabs(["📊 Resultados y Gráficas", "📖 Marco Teórico", "🛡️ Análisis de Seguridad"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Caudal de Fuga (Q)", f"{res['Q']*1000:.4f} L/s/m")
    col2.metric("Gradiente Crítico (ic)", f"{res['ic']:.2f}")
    col3.metric("FS Sifonamiento", f"{res['fs']:.2f}")

    # Gráficas estilo Colab original
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    nx, ny, dx, x_ini, x_fin, y_base, y_sup = res['coords']
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dx)
    
    # Gráfico 1: Red de flujo azul y flechas blancas
    ax1.contourf(X, Y, res['H'], levels=50, cmap='Blues', alpha=0.8)
    ax1.contour(X, Y, res['H'], levels=np.linspace(h2, h1, 15), colors='black', linewidths=0.5)
    ax1.streamplot(X, Y, res['ix'], res['iy'], color='white', density=2.0, linewidth=1.0)
    ax1.imshow(np.where(res['mask'], np.nan, 1), extent=[0, lx, 0, 30.0], origin='lower', cmap='gray', alpha=0.9, zorder=5)
    ax1.set_title(f'Red de Flujo y Líneas de Corriente (Lx = {lx}m, Muro = {prof_muro}m a Pos = {pos_muro}m)')
    ax1.set_ylabel('Elevación (m)')
    
    # Gráfico 2: Mapa de calor Turbo
    mapa_grad = ax2.contourf(X, Y, res['imag'], levels=np.linspace(0, 1.2, 50), cmap='turbo', extend='max')
    plt.colorbar(mapa_grad, ax=ax2, label='Gradiente Hidráulico (i)')
    ax2.imshow(np.where(res['mask'], np.nan, 1), extent=[0, lx, 0, 30.0], origin='lower', cmap='gray', alpha=1.0, zorder=5)
    ax2.contour(X, Y, res['imag'], levels=[res['ic']], colors='red', linewidths=3, linestyles='dashed')
    ax2.set_title('Mapa de Calor del Gradiente Hidráulico')
    ax2.set_xlabel('Posición X (m)')
    ax2.set_ylabel('Elevación (m)')

    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0, lx)
        ax.set_ylim(0, 30.0)

    plt.tight_layout()
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
    st.markdown(f"**Gradiente Máximo de Salida Calculado:** {res['i_exit_max']:.3f}")
    if res['fs'] > 1.5:
        st.success(f"Sistema Seguro. FS = {res['fs']:.2f} (> 1.5)")
    elif res['fs'] > 1.0:
        st.warning(f"Condición Crítica. FS = {res['fs']:.2f} (Margen normativo insuficiente)")
    else:
        st.error(f"Falla por Sifonamiento Detectada. FS = {res['fs']:.2f} (<= 1.0)")
