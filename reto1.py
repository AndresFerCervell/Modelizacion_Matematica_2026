"""
Reto 1 - "Más vale prevenir": colocación óptima de k cortafuegos (s = 2 etapas).

Estructura:
    - cargar_prob_inicio    : lee inicio.txt
    - matriz_adyacencia     : DiGraph -> matriz de probabilidades P
    - esperanza_quemadas    : evalúa E[|Q_2|] para una matriz P (función objetivo exacta)
    - fuerza_bruta          : búsqueda exhaustiva de las C(|A|, k) combinaciones
    - branch_and_bound      : ramificación y poda exacta (cota de ganancias)
    - resolver              : elige método según el nº de aristas (UMBRAL_ARISTAS)
    - verificar_montecarlo  : comprueba el resultado con el simulador de engine.py
"""
import glob
import itertools
import os
import sys
import time

import numpy as np
import networkx as nx

from core import leer_grafo
from engine import IncendioSimulator

# ----------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------
K_CORTAFUEGOS = 4
S_ETAPAS = 2

# A partir de este nº de aristas se usa Branch & Bound en lugar de fuerza bruta.
# C(50, 4) = 230.300 combinaciones (segundos); C(80, 4) = 1.58M (minutos).
# Ajustar tras medir tiempos reales.
UMBRAL_ARISTAS = 6


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def cargar_prob_inicio(ruta_archivo):
    """
    Lee inicio.txt. Formato esperado por línea: 'nodo probabilidad'.
    Devuelve dict {nodo: prob}. Soporta comas decimales y líneas vacías.
    """
    probs = {}
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            if len(partes) >= 2:
                probs[int(partes[0])] = float(partes[1].replace(",", "."))
    return probs


def matriz_adyacencia(G):
    """Matriz P (n x n) con P[i, j] = p_ij si existe el arco (i, j), 0 si no."""
    n = G.n = G.number_of_nodes()
    P = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        P[u, v] = d["weight"]
    return P


def vector_inicio(G):
    """Vector p_l de probabilidades de inicio (guardado en los nodos por leer_grafo)."""
    n = G.n = G.number_of_nodes()
    return np.array([G.nodes[i].get("prob_inicio", 0.0) for i in range(n)])


# ----------------------------------------------------------------------------
# Función objetivo (exacta para s = 2)
# ----------------------------------------------------------------------------
def esperanza_quemadas(P, p_ini):
    """
    E[|Q_2|] = sum_l p_l * ( 1 + sum_{m != l} (1 - S_lm) )

    con S_lm = (1 - P[l,m]) * prod_j (1 - P[l,j] * P[j,m])

    Los caminos l->m y l->j->m (j distintos) son aristadisjuntos, por lo que
    la independencia es exacta y la fórmula coincide con el simulador.
    Los cortafuegos se aplican poniendo a 0 la entrada correspondiente de P
    (equivale a x_ij = 0). Los términos j = l o j = m valen 1 porque
    P[l,l] = P[m,m] = 0.
    """
    directo = 1.0 - P                                        # (n, n)
    via_j = np.prod(1.0 - P[:, :, None] * P[None, :, :], axis=1)   # (n, n)
    prob_quemado = 1.0 - directo * via_j
    np.fill_diagonal(prob_quemado, 1.0)                      # la zona inicial arde siempre
    return float(p_ini @ prob_quemado.sum(axis=1))


# ----------------------------------------------------------------------------
# Fuerza bruta
# ----------------------------------------------------------------------------
def fuerza_bruta(G, p_ini, k=K_CORTAFUEGOS):
    """
    Evalúa todas las combinaciones de k aristas y devuelve la mejor.

    Returns:
        dict con 'aristas_cortadas', 'esperanza_quemadas', 'esperanza_salvadas',
        'esperanza_sin_cortes', 'combinaciones_evaluadas'.
    """
    n = G.number_of_nodes()

    P = matriz_adyacencia(G)
    aristas = sorted(G.edges())
    k_efectivo = min(k, len(aristas))

    sin_cortes = esperanza_quemadas(P, p_ini)
    mejor_valor = np.inf
    mejor_combo = ()
    evaluadas = 0

    for combo in itertools.combinations(aristas, k_efectivo):
        # Aplicar cortes (x_ij = 0), guardando los pesos para restaurarlos
        pesos = [P[u, v] for u, v in combo]
        for u, v in combo:
            P[u, v] = 0.0

        valor = esperanza_quemadas(P, p_ini)
        evaluadas += 1
        if valor < mejor_valor - 1e-12:
            mejor_valor, mejor_combo = valor, combo

        for (u, v), w in zip(combo, pesos):
            P[u, v] = w

    return {
        "aristas_cortadas": list(mejor_combo),
        "esperanza_quemadas": mejor_valor,
        "esperanza_salvadas": n - mejor_valor,
        "esperanza_sin_cortes": sin_cortes,
        "combinaciones_evaluadas": evaluadas,
    }


# ----------------------------------------------------------------------------
# Branch & Bound
# ----------------------------------------------------------------------------
def branch_and_bound(G, p_ini, k=K_CORTAFUEGOS):
    """
    Búsqueda exacta por ramificación y poda.

    Esquema:
        - Árbol binario: en cada nivel se decide cortar / no cortar una arista
          (ordenadas, p. ej., por impacto decreciente).
        - Cota inferior (_cota_ganancias): daño actual menos la suma de los r
          mayores "techos" de ganancia entre las aristas aún por decidir.
        - Poda: si la cota >= mejor solución incumbente, se descarta la rama.
        - Incumbente inicial: solución greedy.

    Debe devolver el mismo diccionario que `fuerza_bruta`.
    """
    n = G.number_of_nodes()
    
    P = matriz_adyacencia(G)
    sin_cortes = esperanza_quemadas(P, p_ini)

    # 1) Candidatas: solo aristas cuyo corte individual reduce el daño.
    #    Una arista con ganancia 0 en el grafo completo seguirá teniéndola 0
    #    tras otros cortes (cortar solo reduce probabilidades de uso), así que
    #    descartarla no pierde el óptimo.
    ganancias = []
    for (u, v) in sorted(G.edges()):
        w = P[u, v]
        P[u, v] = 0.0
        ganancias.append((sin_cortes - esperanza_quemadas(P, p_ini), (u, v)))
        P[u, v] = w
    ganancias.sort(key=lambda t: -t[0])          # mayor impacto primero
    candidatas = [e for g, e in ganancias if g > 1e-12]
    k_efectivo = min(k, len(candidatas))

    # 2) Incumbente inicial: solución greedy (da una buena cota superior).
    combo_ini = _greedy(P, p_ini, candidatas, k_efectivo)
    P_tmp = P.copy()
    for (u, v) in combo_ini:
        P_tmp[u, v] = 0.0
    valor_ini = esperanza_quemadas(P_tmp, p_ini)
    mejor = {"valor": valor_ini, "combo": tuple(combo_ini)}

    # 3) Búsqueda en profundidad: en el nivel i se decide cortar / no cortar
    #    la arista candidatas[i].
    permitidas = np.zeros((n, n), dtype=bool)    # aristas aún por decidir
    for (u, v) in candidatas:
        permitidas[u, v] = True
    cortes = []
    nodos = 0

    def dfs(i, r):
        nonlocal nodos
        nodos += 1
        if r == 0:                               # hoja: k cortes decididos
            valor = esperanza_quemadas(P, p_ini)
            if valor < mejor["valor"] - 1e-12:
                mejor["valor"], mejor["combo"] = valor, tuple(cortes)
            return
        if len(candidatas) - i < r:              # no quedan aristas suficientes
            return
        # Poda: si ni en el mejor caso (cota de ganancias) se mejora el incumbente
        if _cota_ganancias(P, permitidas, p_ini, r) >= mejor["valor"] - 1e-12:
            return                               # poda

        u, v = candidatas[i]
        w = P[u, v]
        permitidas[u, v] = False
        # Rama 1: cortar (u, v)
        P[u, v] = 0.0
        cortes.append((u, v))
        dfs(i + 1, r - 1)
        cortes.pop()
        P[u, v] = w
        # Rama 2: no cortar (u, v)
        dfs(i + 1, r)
        permitidas[u, v] = True

    dfs(0, k_efectivo)

    # 4) Si hay menos de k candidatas útiles, completar con aristas irrelevantes
    combo = list(mejor["combo"])
    if len(combo) < k:
        resto = [e for e in sorted(G.edges()) if e not in combo]
        combo += resto[:k - len(combo)]

    return {
        "aristas_cortadas": combo,
        "esperanza_quemadas": mejor["valor"],
        "esperanza_salvadas": n - mejor["valor"],
        "esperanza_sin_cortes": sin_cortes,
        "combinaciones_evaluadas": nodos,        # nodos del árbol explorados
    }


def _greedy(P, p_ini, candidatas, k):
    """Elige k aristas una a una, cada vez la que más reduce E[|Q_2|]."""
    P = P.copy()
    elegidas = []
    for _ in range(k):
        mejor_e, mejor_v = None, np.inf
        for e in candidatas:
            if e in elegidas:
                continue
            w = P[e]
            P[e] = 0.0
            v = esperanza_quemadas(P, p_ini)
            P[e] = w
            if v < mejor_v:
                mejor_e, mejor_v = e, v
        P[mejor_e] = 0.0
        elegidas.append(mejor_e)
    return elegidas


def _cota_ganancias(P, permitidas, p_ini, r):
    """
    Cota inferior por "ganancia máxima por arista" (reparte el presupuesto de
    r cortes entre las aristas del grafo).

    Para cada término (l, m), la ganancia de eliminar una vía de probabilidad v es
        v * prod(1 - v_k de las demás vías)  <=  v
    en cualquier contexto de otros cortes (v solo puede bajar al cortar más).
    Sumando sobre los términos donde aparece la arista (a, b):
        D[a,b] = P[a,b] * ( p_a                          # vía directa (a -> b)
                          + sum_{l != b} p_l P[l,a]      # b como 2º salto: l -> a -> b
                          + p_a * sum_{m != a} P[b,m] )  # b como 1er salto: a -> b -> m
    es una cota superior de lo que esa arista puede reducir en cualquier contexto.
    Por telescopía:  f(cortes + S) >= f(cortes) - sum_{e in S} D[e],
    y con |S| <= r basta restar las r mayores D entre las aristas permitidas.
    """
    s_in = p_ini @ P                              # s_in[a] = sum_l p_l P[l,a]
    filas = P.sum(axis=1)                         # filas[b] = sum_m P[b,m]
    D = P * (p_ini[:, None] + s_in[:, None] - p_ini[None, :] * P.T
             + p_ini[:, None] * (filas[None, :] - P.T))
    gan = D[permitidas]
    if gan.size > r:
        gan = np.partition(gan, -r)[-r:]
    return esperanza_quemadas(P, p_ini) - float(gan.sum())


# ----------------------------------------------------------------------------
# Selector de método
# ----------------------------------------------------------------------------
def resolver(G, p_ini, k=K_CORTAFUEGOS, umbral=UMBRAL_ARISTAS):
    """Fuerza bruta si |A| <= umbral; Branch & Bound en caso contrario."""
    if G.number_of_edges() <= umbral:
        return fuerza_bruta(G, p_ini, k)
    return branch_and_bound(G, p_ini, k)


# ----------------------------------------------------------------------------
# Verificación con el simulador (Monte Carlo)
# ----------------------------------------------------------------------------

def verificar_montecarlo(G, p_ini, aristas_cortadas, s=S_ETAPAS,
                         n_sim=20000, semilla=0):
    """
    Estima por simulación el nº medio de zonas incendiadas en s etapas usando
    IncendioSimulator. Sirve para validar la fórmula exacta.
    """
    rng = np.random.default_rng(semilla)
    p = np.asarray(p_ini, dtype=float)
    p = p / p.sum()
    total = 0
    for i in range(n_sim):
        sim = IncendioSimulator(G, semilla=int(rng.integers(1 << 31)))
        for u, v in aristas_cortadas:
            sim.bomberos_cortar_arista(u, v)
        sim.iniciar_incendio(int(rng.choice(len(p), p=p)))
        for _ in range(s):
            if not sim.pasar_turno():
                break
        total += len(sim.Q_t)
    return total / n_sim


# ----------------------------------------------------------------------------
# Ejecución sobre los ficheros del reto
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Carpeta de datos: por defecto "data/"; se puede pasar otra como argumento:
    #     python optimizador.py otra_carpeta
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "data"
    prob_ini = cargar_prob_inicio(os.path.join(carpeta, "inicio.txt"))

    # Procesa todos los ficheros graph_*.txt de la carpeta (p. ej. graph_003_probs.txt)
    for ruta in sorted(glob.glob(os.path.join(carpeta, "graph_*.txt"))):
        G = leer_grafo(ruta, probabilidades_inicio=prob_ini)
        p_ini = vector_inicio(G)

        t0 = time.time()
        res = resolver(G, p_ini)
        dt = time.time() - t0

        print(f"\n=== {ruta} | {G.number_of_edges()} aristas | {dt:.2f}s ===")
        print(f"Aristas cortadas        : {res['aristas_cortadas']}")
        print(f"E[zonas incendiadas]    : {res['esperanza_quemadas']:.4f} "
              f"(sin cortes: {res['esperanza_sin_cortes']:.4f})")
        print(f"E[zonas salvadas]       : {res['esperanza_salvadas']:.4f}")

        mc = verificar_montecarlo(G, p_ini, res["aristas_cortadas"], n_sim=5000)
        print(f"Comprobación Monte Carlo: {mc:.4f}")