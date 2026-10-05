"""
Genera los datos SINTÉTICOS del Reto 5 a partir de los grafos del concurso.

El enunciado no proporciona datos para el Reto 5, así que los valores de zonas,
multiplicadores y efectivos son inventados (con semilla fija, por lo que el
resultado es siempre el mismo). Lo que se defiende es el método, no estos números.

Para cada fichero data/graph_XXX*.txt crea en data_reto5/:
    aristas_XXX.txt    origen destino prob multiplicador
    nodos_XXX.txt      nodo tipo personas biodiversidad velocidad
    efectivos_XXX.txt  tipo fuerza duracion t_listo t_recarga rango base

Uso:
    python generar_datos_reto5.py                      # data/ -> data_reto5/
    python generar_datos_reto5.py origen destino       # otras carpetas
"""
import glob
import os
import re
import sys

import networkx as nx
import numpy as np

SEMILLA_BASE = 2026
NODO_INICIO = 1          # zona donde empieza el incendio (q = 1 en los Retos 2-3)
NODOS_MINIMOS = 20       # las zonas de los grafos del concurso van de 0 a 19

# Tipos de zona: rangos (min, max) de personas y biodiversidad en % (0-100, no suman
# 100), velocidad de quemado (fracción del nodo por hora) y proporción de zonas.
TIPOS = {
    #            personas      biodiversidad  velocidad       proporción
    "ciudad":  ((85, 100),     (0, 5),        (0.15, 0.25),   0.10),
    "pueblo":  ((40, 70),      (10, 30),      (0.25, 0.40),   0.20),
    "reserva": ((0, 5),        (70, 100),     (0.45, 0.65),   0.15),
    "bosque":  ((0, 10),       (30, 60),      (0.45, 0.65),   0.25),
    "secano":  ((0, 5),        (0, 10),       (0.60, 0.85),   0.30),
}

# Multiplicador de aristas (meteorología, anchura de la frontera): N(1, 0.25) en [0.5, 1.5]
MULT_MEDIA, MULT_DESV, MULT_MIN, MULT_MAX = 1.0, 0.25, 0.5, 1.5

# Efectivos: (cantidad, fuerza, duracion, t_listo, t_recarga, rango) con variación +-10 %
TERRESTRES = dict(cantidad=3, fuerza=0.5, duracion=0.0, t_listo=5.0, t_recarga=0.0, rango=2)
AEREOS = dict(cantidad=2, fuerza=1.5, duracion=1.0, t_listo=2.5, t_recarga=2.0, rango=-1)


def leer_aristas(ruta):
    aristas = []
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            t = linea.strip().replace(",", ".").split()
            if len(t) >= 3 and not linea.startswith("#"):
                aristas.append((int(t[0]), int(t[1]), float(t[2])))
    return aristas


def reparto_tipos(n, rng):
    """Asigna un tipo a cada zona respetando (aprox.) las proporciones."""
    nombres = list(TIPOS)
    exacto = np.array([TIPOS[k][3] * n for k in nombres])
    cuenta = np.floor(exacto).astype(int)
    for i in np.argsort(-(exacto - cuenta))[: n - cuenta.sum()]:
        cuenta[i] += 1
    lista = [k for k, c in zip(nombres, cuenta) for _ in range(c)]
    rng.shuffle(lista)
    return lista


def uniforme(rng, rango, decimales=2):
    return round(float(rng.uniform(*rango)), decimales)


def escribir_aristas(ruta, aristas, rng):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# origen destino prob multiplicador\n")
        f.write("# prob: probabilidad de que la arista transmita (la del grafo original)\n")
        f.write("# multiplicador: factor de velocidad del frente (meteorología, anchura...)\n")
        for u, v, p in aristas:
            m = float(np.clip(rng.normal(MULT_MEDIA, MULT_DESV), MULT_MIN, MULT_MAX))
            f.write(f"{u} {v} {p} {m:.2f}\n")


def escribir_nodos(ruta, n, rng):
    tipos = reparto_tipos(n, rng)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# nodo tipo personas biodiversidad velocidad\n")
        f.write("# personas y biodiversidad en % (0-100, no tienen que sumar 100)\n")
        f.write("# velocidad: fracción de la zona que se quema por hora\n")
        for i, tipo in enumerate(tipos):
            pers, bio, vel, _ = TIPOS[tipo]
            f.write(f"{i} {tipo} {uniforme(rng, pers, 0):.0f} {uniforme(rng, bio, 0):.0f} "
                    f"{uniforme(rng, vel):.2f}\n")


def escribir_efectivos(ruta, aristas, n, rng):
    # Bases de los terrestres: zonas a 1-3 saltos del foco, para que puedan intervenir
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from((u, v) for u, v, _ in aristas)
    dist = nx.single_source_shortest_path_length(G, NODO_INICIO)
    cercanas = [v for v, d in dist.items() if 1 <= d <= 3]
    otras = [v for v in range(n) if v != NODO_INICIO and v not in cercanas]
    bases = list(rng.permutation(cercanas)) + list(rng.permutation(otras))
    bases = [int(b) for b in bases[: TERRESTRES["cantidad"]]]

    def var(x):
        return round(x * float(rng.uniform(0.9, 1.1)), 2)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# tipo fuerza duracion t_listo t_recarga rango base\n")
        f.write("# fuerza: fracción de zona/hora que anula al fuego\n")
        f.write("# duracion: horas que dura una descarga (solo aéreos)\n")
        f.write("# t_listo: horas hasta estar listo tras el inicio del incendio\n")
        f.write("# t_recarga: horas entre descargas (solo aéreos)\n")
        f.write("# rango: saltos máximos desde la base (-1 = sin límite)\n")
        f.write("# base: zona de la estación base (-1 = ninguna)\n")
        for base in bases:
            T = TERRESTRES
            f.write(f"terrestre {var(T['fuerza'])} {T['duracion']} {var(T['t_listo'])} "
                    f"{T['t_recarga']} {T['rango']} {base}\n")
        for _ in range(AEREOS["cantidad"]):
            A = AEREOS
            f.write(f"aereo {var(A['fuerza'])} {A['duracion']} {var(A['t_listo'])} "
                    f"{var(A['t_recarga'])} {A['rango']} -1\n")


def generar(origen="data", destino="data_reto5"):
    os.makedirs(destino, exist_ok=True)
    ficheros = sorted(glob.glob(os.path.join(origen, "graph_*.txt")))
    if not ficheros:
        print(f"No se encontró ningún graph_*.txt en '{origen}'")
        return []
    generados = []
    for ruta in ficheros:
        num = re.search(r"graph_(\d+)", os.path.basename(ruta)).group(1)
        aristas = leer_aristas(ruta)
        n = max(NODOS_MINIMOS, 1 + max(max(u, v) for u, v, _ in aristas))
        rng = np.random.default_rng([SEMILLA_BASE, int(num)])
        escribir_aristas(os.path.join(destino, f"aristas_{num}.txt"), aristas, rng)
        escribir_nodos(os.path.join(destino, f"nodos_{num}.txt"), n, rng)
        escribir_efectivos(os.path.join(destino, f"efectivos_{num}.txt"), aristas, n, rng)
        print(f"graph_{num}: {n} zonas, {len(aristas)} aristas -> {destino}/")
        generados.append(num)
    return generados


if __name__ == "__main__":
    origen = sys.argv[1] if len(sys.argv) > 1 else "data"
    destino = sys.argv[2] if len(sys.argv) > 2 else "data_reto5"
    generar(origen, destino)