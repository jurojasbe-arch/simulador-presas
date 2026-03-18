import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(layout="wide", page_title="Simulador de Flujo 2D")
st.title("Modelación Interactiva de Flujo y Sifonamiento bajo Presas")
st.markdown("Ajusta los parámetros en el panel lateral para calcular la red de flujo y los gradientes en tiempo real.")

# --- 2. PANEL LATERAL (CONTROLES INTERACTIVOS) ---
st.sidebar.header("Parámetros Geométricos")
Lx_val = st.sidebar.slider("Dominio Horizontal (Lx) [m]", min_value=75.0, max_value=150.0, value=135.0, step=5.0)
prof_muro_val = st.sidebar.slider("Profundidad del Muro [m]", min_value=0.0, max_value=25.0, value=10.0, step=1.0)
pos_muro_val = st.sidebar.slider("Posición del Muro (desde aguas arriba) [m]", min_value=0.0, max_value=15.0, value=5.0, step=1.0)

st.sidebar.header("Parámetros Geotécnicos")
Gs_val = st.sidebar.slider("Gravedad Específica (Gs)", 2.50, 2.80, 2.65, step=0.01)
e_val = st.sidebar.slider("Relación de Vacíos (e)", 0.40, 1.00, 0.65, step=0.05)
k_val = st.sidebar.number_input("Permeabilidad (k) [m/s]", value=1e-5, format="%.1e")

st.sidebar.header("Resolución Computacional")
dx_val = st.sidebar.selectbox("Tamaño de Malla (dx) [m]", [1.0, 0.5], index=0, help="1.0m para velocidad, 0.5m para precisión.")

# --- 3. NÚCLEO MATEMÁTICO (FUNCIÓN CACHEADA) ---
@st.cache_data
def resolver_flujo(Lx, prof_muro, pos_muro_x, Gs, e, k, dx):
    Ly, base_presa, empotramiento, h1, h2 = 30.0, 15.0, 5.0, 50.0, 5.0
    Nx, Ny = int(Lx / dx) + 1, int(Ly / dx) + 1
    ic_critico = (Gs - 1) / (1 + e)
    
    x_inicio = (Lx - base_presa) / 2
    idx_x_inicio = int(x_inicio / dx)
    idx_x_fin = int((x_inicio + base_presa) / dx)
    idx_y_sup = Ny - 1  
    idx_y_base = int((Ly - empotramiento) / dx) 
    idx_x_muro = int((x_inicio + pos_muro_x) / dx)
    idx_y_muro_fin = int((Ly - empotramiento - prof_muro) / dx) 
    
    is_soil = np.ones((Ny, Nx), dtype=bool)
    is_soil[idx_y_base:idx_y_sup+1, idx_x_inicio:idx_x_fin+1] = False
    if prof_muro > 0:
        is_soil[idx_y_muro_fin:idx_y_base, idx_x_muro] = False
        
    H = np.zeros((Ny, Nx))
    for i in range(Nx): H[:, i] = h1 - (h1 - h2) * (i / Nx)
    H[idx_y_sup, :idx_x_inicio] = h1
    H[idx_y_sup, idx_x_fin+1:] = h2
    
    calc_mask = is_soil.copy()
    calc_mask[idx_y_sup, :idx_x_inicio] = False 
    calc_mask[idx_y_sup, idx_x_fin+1:] = False  
    
    tolerancia, error, iteraciones, max_iter = 1e-4, 1.0, 0, 8000
    
    while error > tolerancia and iteraciones < max_iter:
        H_viejo = H.copy()
        H_arriba = H[2:, 1:-1].copy()
        H_abajo  = H[:-2, 1:-1].copy()
        H_der    = H[1:-1, 2:].copy()
        H_izq    = H[1:-1, :-2].copy()
        
        H_arriba[~is_soil[2:, 1:-1]] = H[1:-1, 1:-1][~is_soil[2:, 1:-1]]
        H_abajo[~is_soil[:-2, 1:-1]] = H[1:-1, 1:-1][~is_soil[:-2, 1:-1]]
        H_der[~is_soil[1:-1, 2:]]    = H[1:-1, 1:-1][~is_soil[1:-1, 2:]]
        H_izq[~is_soil[1:-1, :-2]]   = H[1:-1, 1:-1][~is_soil[1:-1, :-2]]
        
        H_new = H.copy()
        H_new[1:-1, 1:-1] = 0.25 * (H_arriba + H_abajo + H_der + H_izq)
        H_new[0, 1:-1] = np.where(is_soil[0, 1:-1], 0.25 * (2*H_new[1, 1:-1] + H_new[0, 2:] + H_new[0, :-2]), H_new[0, 1:-1])
        H_new[1:-1, 0] = np.where(is_soil[1:-1, 0], 0.25 * (H_new[2:, 0] + H_new[:-2, 0] + 2*H_new[1:-1, 1]), H_new[1:-1, 0])
        H_new[1:-1, -1]= np.where(is_soil[1:-1, -1],0.25 * (H_new[2:, -1] + H_new[:-2, -1] + 2*H_new[1:-1, -2]), H_new[1:-1, -1])
        
        H_new[idx_y_sup, :idx_x_inicio] = h1
        H_new[idx_y_sup, idx_x_fin+1:] = h2
        H[calc_mask] = H_new[calc_mask]
        
        error = np.max(np.abs(H - H_viejo))
        iteraciones += 1

    iy, ix = np.gradient(-H, dx, dx)
    imag = np.sqrt(ix**2 + iy**2)
    imag[~is_soil] = np.nan
    ix[~is_soil] = 0.0
    iy[~is_soil] = 0.0
    
    Q = np.sum(k * ix[:, idx_x_inicio] * dx) if prof_muro == 0 else np.sum(k * ix[:, idx_x_muro][is_soil[:, idx_x_muro]] * dx)
    
    return H, imag, ix, iy, is_soil, Q, ic_critico

# --- 4. EJECUCIÓN Y TABLERO DE RESULTADOS ---
with st.spinner('Calculando matriz de diferencias finitas...'):
    H, imag, ix, iy, mask, Q, ic = resolver_flujo(Lx_val, prof_muro_val, pos_muro_val, Gs_val, e_val, k_val, dx_val)

col1, col2, col3 = st.columns(3)
col1.metric("Caudal de Infiltración (Q)", f"{Q*1000:.4f} L/s/m")
col2.metric("Gradiente Crítico (ic)", f"{ic:.2f}")
grad_max = np.nanmax(imag)
col3.metric("Gradiente Máximo Calculado", f"{grad_max:.3f}", delta="Peligro!" if grad_max >= ic else "Seguro", delta_color="inverse")

# --- 5. RENDERIZADO DE GRÁFICOS ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
X, Y = np.meshgrid(np.arange(H.shape[1])*dx_val, np.arange(H.shape[0])*dx_val)

ax1.contourf(X, Y, H, levels=40, cmap='Blues', alpha=0.8)
ax1.contour(X, Y, H, levels=np.linspace(5, 50, 15), colors='black', linewidths=0.5)
ax1.streamplot(X, Y, ix, iy, color='white', density=1.5, linewidth=1.0)
ax1.contourf(X, Y, mask, levels=[0, 0.5], colors=['gray'])
ax1.set_title(f"Líneas Equipotenciales y de Corriente (Dominio Lx = {Lx_val}m)")
ax1.set_ylabel("Elevación (m)")

cmap_grad = plt.get_cmap('turbo')
mapa = ax2.contourf(X, Y, imag, levels=np.linspace(0, 1.2, 50), cmap=cmap_grad, extend='max')
fig.colorbar(mapa, ax=ax2, label="Gradiente Hidráulico (i)")
ax2.contourf(X, Y, mask, levels=[0, 0.5], colors=['black'])
ax2.contour(X, Y, imag, levels=[ic], colors='red', linewidths=3, linestyles='dashed')
ax2.set_title("Mapa de Calor: Concentración de Gradientes y Zona Crítica")
ax2.set_xlabel("Posición X (m)")
ax2.set_ylabel("Elevación (m)")

for ax in [ax1, ax2]:
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

st.pyplot(fig)