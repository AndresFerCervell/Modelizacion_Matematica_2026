"""
Pruebas de coherencia del Reto 5. Uso:  python test_reto5.py
(necesita data/graph_003*.txt y haber ejecutado generar_datos_reto5.py)
"""
import glob
import os

import numpy as np

from core import leer_grafo
from engine import IncendioSimulator
from generar_datos_reto5 import leer_aristas
from reto5 import cargar_grafo, comparar, simular_episodio
from reto5_efectivos import AEREO, TERRESTRE, Brigada
from reto5_modelo import (ARDIENDO, EXTINGUIDO, QUEMADO, SANO, Bosque, Incendio,
                          Parametros)
from reto5_politicas import (Politica, PoliticaAleatoria, PoliticaGreedyRiesgo,
                             PoliticaNula)

CARPETA_DATOS = "data_reto5"
NUM = "003"
RUTA_GRAFO = glob.glob(os.path.join("data", "graph_003*.txt"))[0]


def prueba(nombre):
    def deco(f):
        def envoltura():
            f()
            print(f"OK   {nombre}")
        return envoltura
    return deco


@prueba("Caso límite: sin efectivos reproduce el modelo discreto del enunciado (engine.py)")
def test_caso_limite():
    # dt = 1, un nodo arde en un tick (velocidad 1), multiplicador 1 y theta = 0
    aristas = leer_aristas(RUTA_GRAFO)
    u, v, p = zip(*aristas)
    n = 20
    bosque = Bosque(n, u, v, p, [1.0] * len(u), [1.0] * n, [1.0] * n, [1.0] * n)
    params = Parametros(dt=1.0, theta=0.0)
    N = 20000
    nuevo = np.array([simular_episodio(bosque, [], PoliticaNula(), params, s)["nodos_equivalentes"]
                      for s in range(N)])
    G = leer_grafo(RUTA_GRAFO)
    motor = np.array([IncendioSimulator(G, semilla=10_000 + s).simular_episodio(nodo_inicio=1)
                      for s in range(N)])
    dif = abs(nuevo.mean() - motor.mean())
    error = np.sqrt(nuevo.var() / N + motor.var() / N)
    print(f"     nuevo={nuevo.mean():.3f}  engine.py={motor.mean():.3f}  "
          f"diferencia={dif:.3f} (error típico {error:.3f})")
    assert dif < 4 * error, "El caso límite no coincide con el modelo discreto"


@prueba("Reproducibilidad: misma semilla, mismo resultado")
def test_reproducibilidad():
    bosque, plantilla = cargar_grafo(CARPETA_DATOS, NUM)
    a = simular_episodio(bosque, plantilla, PoliticaGreedyRiesgo(), Parametros(), 123)
    b = simular_episodio(bosque, plantilla, PoliticaGreedyRiesgo(), Parametros(), 123)
    assert a == b


@prueba("Lo quemado nunca decrece y un nodo extinguido queda congelado")
def test_monotonia_y_extincion():
    bosque, plantilla = cargar_grafo(CARPETA_DATOS, NUM)
    params = Parametros()
    politica = PoliticaGreedyRiesgo()
    for semilla in range(60):
        inc = Incendio(bosque, params, np.random.default_rng([semilla, 0]))
        rng_ruido = np.random.default_rng([semilla, 1])
        brig = Brigada(plantilla)
        b_prev = inc.b.copy()
        congelado = {}
        while not inc.terminado() and inc.t < params.t_max:
            brig.asignar(politica.decidir(inc.observar(), bosque, brig.efectivos, params), bosque)
            S = brig.supresion(bosque.n, rng_ruido, params.sigma_ruido)
            inc.avanzar(S, params.dt)
            brig.avanzar(params.dt)
            assert (inc.b >= b_prev - 1e-12).all(), "lo quemado ha decrecido"
            assert (inc.b <= 1 + 1e-12).all()
            for v in np.where(inc.estado == EXTINGUIDO)[0]:
                congelado.setdefault(v, inc.b[v])
                assert abs(inc.b[v] - congelado[v]) < 1e-12, "un nodo extinguido ha seguido quemándose"
            b_prev = inc.b.copy()


class _PoliticaVerificada(Politica):
    """Envuelve al greedy y comprueba que cada decisión cumple las restricciones."""
    nombre = "greedy verificado"

    def __init__(self):
        self.interna = PoliticaGreedyRiesgo()
        self.decisiones = 0

    def decidir(self, obs, bosque, efectivos, params):
        asig = self.interna.decidir(obs, bosque, efectivos, params)
        por_id = {e.id: e for e in efectivos}
        for id_e, v in asig.items():
            e = por_id[id_e]
            assert e.libre_para_asignar(), "se asignó un efectivo que no estaba libre"
            assert e.puede_actuar_en(v, bosque), "terrestre fuera de su rango"
            if e.tipo == AEREO:
                assert obs.ritmo_visible[v] > 0, "descarga aérea sin fuego visible"
            self.decisiones += 1
        return asig


@prueba("El greedy respeta rango, disponibilidad y descargas solo sobre fuego visible")
def test_restricciones():
    bosque, plantilla = cargar_grafo(CARPETA_DATOS, NUM)
    pol = _PoliticaVerificada()
    for semilla in range(80):
        simular_episodio(bosque, plantilla, pol, Parametros(), semilla)
    assert pol.decisiones > 0


@prueba("Los efectivos ayudan: greedy < aleatoria < sin efectivos (mismas semillas)")
def test_greedy_mejora():
    bosque, plantilla = cargar_grafo(CARPETA_DATOS, NUM)
    nula, aleat, greedy = comparar(
        bosque, plantilla, [PoliticaNula(), PoliticaAleatoria(1), PoliticaGreedyRiesgo()],
        Parametros(), 300, 99)
    for a, b, nombre in [(nula, greedy, "greedy"), (aleat, greedy, "greedy vs aleatoria"),
                         (nula, aleat, "aleatoria")]:
        dif = a.destruccion - b.destruccion
        print(f"     ahorro {nombre}: {100 * dif.mean():.2f} pts "
              f"(± {196 * dif.std(ddof=1) / np.sqrt(len(dif)):.2f})")
        assert dif.mean() > 2 * dif.std(ddof=1) / np.sqrt(len(dif)), f"{nombre} no mejora"


if __name__ == "__main__":
    for t in (test_caso_limite, test_reproducibilidad, test_monotonia_y_extincion,
              test_restricciones, test_greedy_mejora):
        t()
    print("\nTodas las pruebas superadas.")