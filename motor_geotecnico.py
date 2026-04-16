import numpy as np

# ACTUALIZADA: La función ahora acepta h1, h2 y la posición del muro
def modelar_flujo(Lx, prof_muro, pos_muro, dx=0.5, k=1e-5, Gs=2.65, e=0.65, h1=50.0, h2=5.0):
    # Parámetros fijos de la geometría
    Ly, base_presa, empotramiento = 30.0, 15.0, 5.0
    
    Nx, Ny = int(Lx / dx) + 1, int(Ly / dx) + 1
    ic_critico = (Gs - 1) / (1 + e)
    
    # Índices geométricos
    x_inicio = (Lx - base_presa) / 2
    idx_x_inicio = int(x_inicio / dx)
    idx_x_fin = int((x_inicio + base_presa) / dx)
    idx_y_base = int((Ly - empotramiento) / dx) 
    
    # ACTUALIZADO: Cálculo del índice para la posición del muro
    idx_x_muro = int((x_inicio + pos_muro) / dx)
    idx_y_muro_fin = int((Ly - empotramiento - prof_muro) / dx) 
    
    # Máscara de Concreto (is_soil=False donde hay estructura)
    is_soil = np.ones((Ny, Nx), dtype=bool)
    is_soil[idx_y_base:Ny, idx_x_inicio:idx_x_fin+1] = False
    if prof_muro > 0:
        is_soil[idx_y_muro_fin:idx_y_base, idx_x_muro] = False
        
    # Inicialización de matriz H
    H = np.zeros((Ny, Nx))
    for i in range(Nx): H[:, i] = h1 - (h1 - h2) * (i / Nx)
    
    # Bucle de solución (Gauss-Seidel)
    tolerancia, error, it = 1e-4, 1.0, 0
    while error > tolerancia and it < 10000:
        H_old = H.copy()
        H_new = H.copy()
        
        # Nodos internos con Neumann en bordes de estructura
        # (Se usan promedios de vecinos considerando la máscara is_soil)
        H_up = np.roll(H, -1, axis=0); H_down = np.roll(H, 1, axis=0)
        H_left = np.roll(H, 1, axis=1); H_right = np.roll(H, -1, axis=1)
        
        H_new[1:-1, 1:-1] = 0.25 * (H_up[1:-1, 1:-1] + H_down[1:-1, 1:-1] + 
                                   H_left[1:-1, 1:-1] + H_right[1:-1, 1:-1])
        
        # Neumann Fronteras Exteriores e Impermeables
        H_new[0, 1:-1] = 0.25 * (2*H_new[1, 1:-1] + H_new[0, 2:] + H_new[0, :-2]) # Fondo
        H_new[1:-1, 0] = 0.25 * (H_new[2:, 0] + H_new[:-2, 0] + 2*H_new[1:-1, 1]) # Lateral Izq
        H_new[1:-1, -1]= 0.25 * (H_new[2:, -1] + H_new[:-2, -1] + 2*H_new[1:-1, -2]) # Lateral Der
        
        # CORRECCIÓN DE ESQUINAS (Carga vecinos inmediatos)
        H_new[0, 0] = H_new[1, 0]
        H_new[0, -1] = H_new[1, -1]
        
        # ACTUALIZADO: Restablecer Dirichlet (Cargas conocidas en lechos)
        H_new[Ny-1, :idx_x_inicio] = h1
        H_new[Ny-1, idx_x_fin+1:] = h2
        
        # Solo actualizar donde hay suelo
        H[is_soil] = H_new[is_soil]
        error = np.max(np.abs(H - H_old))
        it += 1

    # Gradientes
    iy, ix = np.gradient(-H, dx, dx)
    i_mag = np.sqrt(ix**2 + iy**2)
    i_mag[~is_soil] = np.nan # Enmascarar concreto en el gradiente
    
    # Caudal Q (Ley de Darcy integrada en columna central)
    idx_central = Nx // 2
    Q = np.sum(k * ix[is_soil[:, idx_central], idx_central] * dx)
    
    # Factor de Seguridad contra Sifonamiento
    # Se extrae el gradiente vertical máximo en la superficie de salida
    i_exit_max = np.max(iy[Ny-1, idx_x_fin+1:])
    fs_sifonamiento = ic_critico / i_exit_max if i_exit_max > 0 else 99.0

    return {
        "H": H, "ix": ix, "iy": iy, "imag": i_mag, "mask": is_soil,
        "Q": Q, "ic": ic_critico, "fs": fs_sifonamiento, "i_exit_max": i_exit_max,
        "coords": (Nx, Ny, dx, idx_x_inicio, idx_x_fin, idx_y_base)
    }
