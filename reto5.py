"""
Reto 5 - "Acercándonos a la realidad": simulación y evaluación.

Uso:
    python generar_datos_reto5.py          # una vez: crea data_reto5/ a partir de data/
    python reto5.py                        # evalúa todos los grafos de data_reto5/
    python reto5.py --sims 1000 --semilla 7
    python reto5.py --traza 003            # cuenta paso a paso un episodio del grafo 003
"""
import argparse
import glob
import os
import re
from dataclasses import dataclass

import numpy as np

from reto5_efectivos import AEREO, Brigada, leer_efectivos
from reto5_modelo import NOMBRE_ESTADO, ARDIENDO, Bosque, Incendio, Parametros
from reto5_politicas import (PoliticaAleatoria, PoliticaGreedyRiesgo, PoliticaNula)


# ----------------------------------------------------------------------------
# Un episodio
# ----------------------------------------------------------------------------
def simular_episodio(bosque, plantilla, politica, params, semilla, traza=None):
    """
    Simula un incendio completo y devuelve su resumen.

    Bucle por tick: la política asigna efectivos -> se calcula la supresión ->
    avanza el fuego -> avanzan los relojes de los efectivos.
    La semilla fija el mundo (aristas vivas) y el ruido de los efectivos, de modo
    que dos políticas con la misma semilla se enfrentan al mismo incendio.
    """
    rng_mundo = np.random.default_rng([semilla, 0])
    rng_ruido = np.random.default_rng([semilla, 1])
    incendio = Incendio(bosque, params, rng_mundo)
    brigada = Brigada(plantilla)

    while not incendio.terminado() and incendio.t < params.t_max - 1e-9:
        obs = incendio.observar()
        asignacion = politica.decidir(obs, bosque, brigada.efectivos, params)
        brigada.asignar(asignacion, bosque)
        S = brigada.supresion(bosque.n, rng_ruido, params.sigma_ruido)
        if traza is not None:
            traza.append(_linea_traza(incendio, brigada, S))
        incendio.avanzar(S, params.dt)
        brigada.avanzar(params.dt)
    return incendio.resumen()


def _linea_traza(incendio, brigada, S):
    ardiendo = [v for v in range(incendio.bosque.n) if incendio.estado[v] == ARDIENDO]
    acciones = [f"{e.tipo[:3]}{e.id}->{e.nodo}" for e in brigada.efectivos if e.nodo is not None]
    return (f"t={incendio.t:5.2f}h  quemado={incendio.b.sum():5.2f}  "
            f"ardiendo={ardiendo}  efectivos={acciones}")


# ----------------------------------------------------------------------------
# Evaluación Monte Carlo con semillas comunes
# ----------------------------------------------------------------------------
@dataclass
class Evaluacion:
    nombre: str
    destruccion: np.ndarray
    personas: np.ndarray
    biodiversidad: np.ndarray
    nodos: np.ndarray


def evaluar(bosque, plantilla, politica, params, semillas):
    res = [simular_episodio(bosque, plantilla, politica, params, s) for s in semillas]
    return Evaluacion(politica.nombre,
                      np.array([r["destruccion"] for r in res]),
                      np.array([r["personas"] for r in res]),
                      np.array([r["biodiversidad"] for r in res]),
                      np.array([r["nodos_equivalentes"] for r in res]))


def comparar(bosque, plantilla, politicas, params, n_sim, semilla):
    """Evalúa todas las políticas sobre los MISMOS incendios (mismas semillas)."""
    semillas = [int(x) for x in np.random.default_rng(semilla).integers(0, 2**31 - 1, n_sim)]
    return [evaluar(bosque, plantilla, pol, params, semillas) for pol in politicas]


def _ic95(x):
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def imprimir_comparacion(titulo, evaluaciones):
    base = evaluaciones[0]                       # la referencia sin efectivos
    print(f"\n=== {titulo} ===")
    print(f"{'política':<22}{'destrucción %':>16}{'personas %':>13}{'biodiv. %':>12}"
          f"{'ahorro vs sin efectivos':>26}")
    for ev in evaluaciones:
        d = 100 * ev.destruccion
        if ev is base:
            ahorro = "-"
        else:
            dif = 100 * (base.destruccion - ev.destruccion)
            ahorro = f"{dif.mean():5.2f} ± {_ic95(dif):.2f} pts"
        print(f"{ev.nombre:<22}{d.mean():>10.2f} ± {_ic95(d):<4.2f}"
              f"{100 * ev.personas.mean():>11.2f}{100 * ev.biodiversidad.mean():>12.2f}"
              f"{ahorro:>26}")


# ----------------------------------------------------------------------------
# Carga de datos y programa principal
# ----------------------------------------------------------------------------
def cargar_grafo(carpeta, num):
    bosque = Bosque.desde_ficheros(os.path.join(carpeta, f"aristas_{num}.txt"),
                                   os.path.join(carpeta, f"nodos_{num}.txt"))
    plantilla = leer_efectivos(os.path.join(carpeta, f"efectivos_{num}.txt"))
    return bosque, plantilla


def numeros_disponibles(carpeta):
    return sorted(re.search(r"aristas_(\d+)", f).group(1)
                  for f in glob.glob(os.path.join(carpeta, "aristas_*.txt")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data_reto5", help="carpeta con los datos del Reto 5")
    ap.add_argument("--sims", type=int, default=500, help="simulaciones por grafo")
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--traza", metavar="NUM", help="cuenta un episodio del grafo NUM")
    args = ap.parse_args()

    params = Parametros()
    numeros = numeros_disponibles(args.datos)
    if not numeros:
        raise SystemExit(f"No hay datos en '{args.datos}'. Ejecuta antes: python generar_datos_reto5.py")

    if args.traza:
        bosque, plantilla = cargar_grafo(args.datos, args.traza)
        traza = []
        res = simular_episodio(bosque, plantilla, PoliticaGreedyRiesgo(), params,
                               args.semilla, traza)
        print("\n".join(traza))
        print({k: round(v, 3) for k, v in res.items()})
        return

    print(f"Zona inicial q = {params.nodo_inicio} | simulaciones por grafo = {args.sims} | "
          f"semilla = {args.semilla}")
    acumulado = None
    for num in numeros:
        bosque, plantilla = cargar_grafo(args.datos, num)
        politicas = [PoliticaNula(), PoliticaAleatoria(args.semilla), PoliticaGreedyRiesgo()]
        evals = comparar(bosque, plantilla, politicas, params, args.sims, args.semilla)
        imprimir_comparacion(f"graph_{num} ({bosque.n} zonas, {len(bosque.u)} aristas)", evals)
        d = np.array([100 * e.destruccion.mean() for e in evals])
        acumulado = d if acumulado is None else acumulado + d

    if len(numeros) > 1:
        print("\n=== Media de los grafos (destrucción ponderada %) ===")
        for nombre, valor in zip(["Sin efectivos", "Aleatoria", "Greedy por riesgo"],
                                 acumulado / len(numeros)):
            print(f"{nombre:<22}{valor:>8.2f}")


if __name__ == "__main__":
    main()