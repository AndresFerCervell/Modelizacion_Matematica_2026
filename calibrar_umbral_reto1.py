"""
Calibración del umbral fuerza bruta vs Branch & Bound.

Genera grafos aleatorios de 20 zonas aumentando el número de aristas de 1 en 1,
mide cuánto tarda cada método (media de varios grafos por tamaño) y determina
a partir de qué número de aristas gana Branch & Bound.

Uso:
    python calibrar_umbral.py                       # valores por defecto
    python calibrar_umbral.py --max-aristas 60      # llegar más lejos
    python calibrar_umbral.py --carpeta data        # usar data/inicio.txt si existe

No necesita añadir nada a reto1.py: reutiliza fuerza_bruta y branch_and_bound.
"""
import argparse
import os
import random
import time
from math import comb

import networkx as nx

import reto1 as op


def probabilidades_inicio(carpeta, n):
    """Usa carpeta/inicio.txt si existe; si no, probabilidad uniforme."""
    ruta = os.path.join(carpeta, "inicio.txt") if carpeta else None
    if ruta and os.path.exists(ruta):
        probs = op.cargar_prob_inicio(ruta)
        return [probs.get(i, 0.0) for i in range(n)], f"{ruta}"
    return [1.0 / n] * n, "uniforme (no se encontró inicio.txt)"


def grafo_aleatorio(m, semilla, p_ini, n):
    """Grafo dirigido aleatorio con n zonas y m aristas, pesos en [0.05, 0.95]."""
    rnd = random.Random(semilla)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, quemado=False, prob_inicio=p_ini[i])
    while G.number_of_edges() < m:
        u, v = rnd.sample(range(n), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=round(rnd.uniform(0.05, 0.95), 4))
    return G


def medir(metodo, G, p):
    t0 = time.perf_counter()
    res = metodo(G, p)
    return time.perf_counter() - t0, res


def calcular_umbral(filas, paso):
    """
    Devuelve (ultimo_m_donde_gana_FB, primer_m_desde_el_que_BB_gana_siempre).
    BB 'gana' en un tamaño si su tiempo medio es menor que el de fuerza bruta.
    """
    gana_bb = [(f["m"], f["t_bb"] < f["t_fb"]) for f in filas]
    primer = None
    for i in range(len(gana_bb)):
        if all(g for _, g in gana_bb[i:]):
            primer = gana_bb[i][0]
            break
    if primer is None:
        return None, None
    ultimo_fb = primer - paso if primer - paso >= 0 else 0
    return ultimo_fb, primer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paso", type=int, default=1, help="incremento de aristas")
    ap.add_argument("--max-aristas", type=int, default=30)
    ap.add_argument("--grafos", type=int, default=3, help="grafos por tamaño")
    ap.add_argument("--carpeta", default="data")
    args = ap.parse_args()

    n = 20
    k = op.K_CORTAFUEGOS
    p_ini, origen_p = probabilidades_inicio(args.carpeta, n)
    print(f"Zonas: {n} | k = {k} | grafos por tamaño: {args.grafos}")
    print(f"Probabilidades de inicio: {origen_p}\n")

    cab = f"{'aristas':>8} {'C(m,k)':>10} {'t fuerza bruta':>15} {'t B&B':>10} {'nodos B&B':>10}  ganador"
    print(cab)
    print("-" * len(cab))

    filas = []
    for m in range(args.paso, args.max_aristas + 1, args.paso):
        t_fb, t_bb, nodos = [], [], []
        for s in range(args.grafos):
            G = grafo_aleatorio(m, 1000 * m + s, p_ini, n)
            p = op.vector_inicio(G)
            tb, rb = medir(op.branch_and_bound, G, p)
            tf, rf = medir(op.fuerza_bruta, G, p)
            # comprobación: ambos métodos deben dar el mismo óptimo
            assert abs(rb["esperanza_quemadas"] - rf["esperanza_quemadas"]) < 1e-9, \
                f"Los métodos difieren con {m} aristas (semilla {s})"
            t_fb.append(tf)
            t_bb.append(tb)
            nodos.append(rb["combinaciones_evaluadas"])
        fila = {
            "m": m,
            "combinaciones": comb(m, k),
            "t_fb": sum(t_fb) / len(t_fb),
            "t_bb": sum(t_bb) / len(t_bb),
            "nodos_bb": sum(nodos) / len(nodos),
        }
        fila["ganador"] = "B&B" if fila["t_bb"] < fila["t_fb"] else "fuerza bruta"
        filas.append(fila)
        print(f"{m:>8} {fila['combinaciones']:>10} {fila['t_fb']:>14.4f}s "
              f"{fila['t_bb']:>9.4f}s {fila['nodos_bb']:>10.0f}  {fila['ganador']}",
              flush=True)

    ultimo_fb, primer_bb = calcular_umbral(filas, args.paso)
    print()
    if primer_bb is None:
        print(f"Con hasta {args.max_aristas} aristas B&B no gana de forma estable: "
              f"prueba con --max-aristas mayor.")
    elif ultimo_fb == 0:
        print("B&B es más rápido ya desde el primer tamaño probado.")
        print("UMBRAL_ARISTAS recomendado = 0  (usar siempre Branch & Bound)")
    else:
        print(f"La fuerza bruta gana hasta {ultimo_fb} aristas; "
              f"B&B gana desde {primer_bb} (y en todos los tamaños mayores probados).")
        print(f"UMBRAL_ARISTAS recomendado = {ultimo_fb}")
    print(f"(resolución de la medida: ±{args.paso} aristas)")

if __name__ == "__main__":
    main()
