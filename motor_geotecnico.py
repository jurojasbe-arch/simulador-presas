import numpy as np

def modelar_flujo(Lx, prof_muro, pos_muro_x, dx=0.5, k=1e-5, Gs=2.65, e=0.65, h1=50.0, h2=5.0):
    # Parámetros fijos
    Ly, base_presa, empotramiento = 30.0, 15.0, 5.0
    
    Nx, Ny = int(Lx / dx) + 1, int(Ly / dx) + 1
    ic_critico = (Gs - 1) / (1 + e)
    
    # Índices geométricos
    x_inicio = (Lx - base_presa) / 2
    idx_x_inicio = int(x_inicio / dx)
    idx_x_fin = int((x_inicio + base_presa) / dx)
    idx_y_sup = Ny - 1  
    idx_y_base = int((Ly - empotramiento) / dx) 
    idx_x_muro = int((x_inicio + pos_muro_x) / dx)
    idx_y_muro_fin = int((Ly - empotramiento - prof_muro) / dx) 
    
    # Máscara (Concreto vs Suelo)
    is_soil = np.ones((Ny, Nx), dtype=bool)
    is_soil[idx_y_base:idx_y_sup+1, idx_x_inicio:idx_x_fin+1] = False
    if prof_muro > 0:
        is_soil[idx_y_muro_fin:idx_y_base, idx_x_muro] = False
        
    # Inicialización
    H = np.zeros((Ny, Nx))
    for i in range(Nx): H[:, i] = h1 - (h1 - h2) * (i / Nx)
    H[idx_y_sup, :idx_x_inicio] = h1
    H[idx_y_sup, idx_x_fin+1:] = h2
    
    calc_mask = is_soil.copy()
    calc_mask[idx_y_sup, :idx_x_inicio] = False 
    calc_mask[idx_y_sup, idx_x_fin+1:] = False  
    
    tolerancia, error, iteraciones = 1e-4, 1.0, 0
    
    # Bucle de solución
    while error > tolerancia and iteraciones < 15000:
        H_viejo = H.copy()
        
        H_arriba, H_abajo = H[2:, 1:-1].copy(), H[:-2, 1:-1].copy()
        H_der, H_izq = H[1:-1, 2:].copy(), H[1:-1, :-2].copy()
        
        # --- LÍNEAS CRÍTICAS RESTAURADAS: Neumann en muros ---
        H_arriba[~is_soil[2:, 1:-1]] = H[1:-1, 1:-1][~is_soil[2:, 1:-1]]
        H_abajo[~is_soil[:-2, 1:-1]] = H[1:-1, 1:-1][~is_soil[:-2, 1:-1]]
        H_der[~is_soil[1:-1, 2:]]    = H[1:-1, 1:-1][~is_soil[1:-1, 2:]]
        H_izq[~is_soil[1:-1, :-2]]   = H[1:-1, 1:-1][~is_soil[1:-1, :-2]]
        # -------------------------------------------------------

        H_new = H.copy()
        H_new[1:-1, 1:-1] = 0.25 * (H_arriba + H_abajo + H_der + H_izq)
        
        # Neumann Fronteras Exteriores
        H_new[0, 1:-1] = np.where(is_soil[0, 1:-1], 0.25 * (2*H_new[1, 1:-1] + H_new[0, 2:] + H_new[0, :-2]), H_new[0, 1:-1])
        H_new[1:-1, 0] = np.where(is_soil[1:-1, 0], 0.25 * (H_new[2:, 0] + H_new[:-2, 0] + 2*H_new[1:-1, 1]), H_new[1:-1, 0])
        H_new[1:-1, -1]= np.where(is_soil[1:-1, -1],0.25 * (H_new[2:, -1] + H_new[:-2, -1] + 2*H_new[1:-1, -2]), H_new[1:-1, -1])
        
        # Corrección de esquinas
        H_new[0, 0] = H_new[1, 0]
        H_new[0, -1] = H_new[1, -1]
        
        H_new[idx_y_sup, :idx_x_inicio] = h1
        H_new[idx_y_sup, idx_x_fin+1:] = h2
        H[calc_mask] = H_new[calc_mask]
        
        error = np.max(np.abs(H - H_viejo))
        iteraciones += 1

    iy, ix = np.gradient(-H, dx, dx)
    imag = np.sqrt(ix**2 + iy**2)
    
    # Máscaras de dibujo para que las flechas rodeen el muro
    ix_enmascarado = np.ma.masked_array(ix, mask=~is_soil)
    iy_enmascarado = np.ma.masked_array(iy, mask=~is_soil)
    imag[~is_soil] = np.nan
    
    # Caudal
    flujo_horizontal = ix[:, idx_x_inicio] if prof_muro == 0 else ix[:, idx_x_muro]
    columna_suelo = is_soil[:, idx_x_inicio] if prof_muro == 0 else is_soil[:, idx_x_muro]
    Q = np.sum(k * flujo_horizontal[columna_suelo] * dx)

    # Variables para Análisis de Seguridad
    i_exit_max = np.max(iy[idx_y_sup, idx_x_fin+1:])
    fs_sifonamiento = ic_critico / i_exit_max if i_exit_max > 0 else 99.0
    
    return {
        "H": H, "imag": imag, "ix": ix_enmascarado, "iy": iy_enmascarado, 
        "mask": is_soil, "Q": Q, "ic": ic_critico, "fs": fs_sifonamiento,
        "i_exit_max": i_exit_max, # Se añade para evitar el KeyError
        "coords": (Nx, Ny, dx, idx_x_inicio, idx_x_fin, idx_y_base, idx_y_sup)
    }
