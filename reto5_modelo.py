"""
Reto 5 - "Acercándonos a la realidad": modelo del bosque y dinámica del incendio.

Ideas del modelo (ver también pseudocodigo / memoria):

  * Tiempo con paso pequeño dt (horas). Cada nodo tiene una fracción quemada
    b_v en [0, 1] que solo crece ("lo quemado, quemado está").
  * Cada arista (u, v) se sortea UNA sola vez al empezar: viva con prob. p_uv o
    muerta. Así el tiempo solo decide CUÁNDO pasa el fuego, nunca SI pasa
    (con un dado por tick, la probabilidad acumulada tendería a 1).
  * Una arista viva se ACTIVA cuando su origen, en llamas, supera la fracción
    theta de quemado. Una vez activa alimenta al destino a ritmo
    velocidad_v * multiplicador_uv hasta que el destino se consume. Varios
    frentes sobre un mismo nodo suman sus ritmos.
  * Los efectivos aplican una supresión S_v sobre un nodo. Ritmo neto =
    max(0, ritmo del fuego - S_v). Si la supresión iguala o supera al fuego
    durante t_extincion horas seguidas, el nodo queda EXTINGUIDO: congelado
    para siempre con lo que lleve quemado (ya no se quema ni transmite más).
  * El incendio termina cuando no queda ningún frente que alimente a un nodo
    aún abierto (SANO o ARDIENDO).
"""
from dataclasses import dataclass

import networkx as nx
import numpy as np

# Estados de un nodo
SANO, ARDIENDO, QUEMADO, EXTINGUIDO = 0, 1, 2, 3
NOMBRE_ESTADO = {SANO: "sano", ARDIENDO: "ardiendo", QUEMADO: "quemado",
                 EXTINGUIDO: "extinguido"}

EPS = 1e-9


# ----------------------------------------------------------------------------
# Parámetros globales
# ----------------------------------------------------------------------------
@dataclass
class Parametros:
    dt: float = 0.25            # paso de tiempo (horas)
    theta: float = 0.5          # fracción quemada a partir de la cual un nodo contagia
    t_extincion: float = 1.0    # horas seguidas con supresion >= fuego para extinguir
    t_max: float = 48.0         # tiempo máximo de seguridad (horas)
    alpha: float = 3.0          # peso del valor humano en el valor de un nodo
    beta: float = 1.0           # peso del valor medioambiental
    gamma: float = 0.5          # peso del valor "aguas abajo" en el riesgo
    sigma_ruido: float = 0.2    # variación aleatoria de la fuerza de cada efectivo (+-)
    nodo_inicio: int = 1        # zona donde empieza el incendio (q)


# ----------------------------------------------------------------------------
# Datos del bosque
# ----------------------------------------------------------------------------
def suma_por_indice(indices, pesos, n):
    """
    out[i] = suma de los pesos cuyo índice es i (array de n decimales).
    Se evita np.bincount a secas: con entradas vacías devuelve enteros y
    truncaría los valores decimales que se sumen después.
    """
    return np.bincount(indices, weights=pesos, minlength=n).astype(float)


def _lineas_utiles(ruta):
    """Líneas no vacías y sin comentarios (#), con comas decimales admitidas."""
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea and not linea.startswith("#"):
                yield linea.replace(",", ".").split()


class Bosque:
    """
    Grafo dirigido con los datos de cada zona y de cada arista.

    Aristas (arrays de longitud E): u (origen), v (destino), p (probabilidad de
    que la arista transmita), m (multiplicador de velocidad: meteorología,
    anchura de la frontera...).
    Nodos (arrays de longitud n): personas y biodiversidad (0-100, no tienen
    por qué sumar 100) y velocidad (fracción del nodo que se quema por hora).
    """

    def __init__(self, n, u, v, p, m, personas, biodiversidad, velocidad,
                 tipos=None):
        self.n = int(n)
        self.u = np.asarray(u, dtype=int)
        self.v = np.asarray(v, dtype=int)
        self.p = np.asarray(p, dtype=float)
        self.m = np.asarray(m, dtype=float)
        self.personas = np.asarray(personas, dtype=float)
        self.biodiversidad = np.asarray(biodiversidad, dtype=float)
        self.velocidad = np.asarray(velocidad, dtype=float)
        self.tipos = list(tipos) if tipos is not None else [""] * self.n
        self.saltos = self._calcular_saltos()

    @classmethod
    def desde_ficheros(cls, ruta_aristas, ruta_nodos):
        """
        aristas: 'origen destino prob multiplicador'
        nodos  : 'nodo tipo personas biodiversidad velocidad'
        """
        u, v, p, m = [], [], [], []
        for t in _lineas_utiles(ruta_aristas):
            u.append(int(t[0])); v.append(int(t[1]))
            p.append(float(t[2])); m.append(float(t[3]))

        filas = sorted(_lineas_utiles(ruta_nodos), key=lambda t: int(t[0]))
        n = len(filas)
        if [int(t[0]) for t in filas] != list(range(n)):
            raise ValueError(f"{ruta_nodos}: los nodos deben ser 0..{n - 1} sin huecos")
        if max(max(u), max(v)) >= n:
            raise ValueError("Hay aristas con extremos que no figuran en el fichero de nodos")
        return cls(n, u, v, p, m,
                   personas=[float(t[2]) for t in filas],
                   biodiversidad=[float(t[3]) for t in filas],
                   velocidad=[float(t[4]) for t in filas],
                   tipos=[t[1] for t in filas])

    def valor(self, params):
        """Valor de cada nodo: mucho más peso a las personas que al medio ambiente."""
        return params.alpha * self.personas + params.beta * self.biodiversidad

    def _calcular_saltos(self):
        """Distancia en saltos entre zonas (ignorando el sentido de las aristas)."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(zip(self.u.tolist(), self.v.tolist()))
        d = np.full((self.n, self.n), np.inf)
        for a, dist in nx.all_pairs_shortest_path_length(G):
            for b, k in dist.items():
                d[a, b] = k
        return d


# ----------------------------------------------------------------------------
# Lo que ven los bomberos
# ----------------------------------------------------------------------------
@dataclass
class Observacion:
    """
    Estado OBSERVABLE del incendio. No incluye qué aristas están vivas pero aún
    sin activar (eso es azar futuro que nadie puede ver); sí incluye los frentes
    ya activos, porque son fuego real que los bomberos ven avanzar.
    """
    t: float
    b: np.ndarray               # fracción quemada de cada nodo
    estado: np.ndarray          # estado de cada nodo
    ritmo_visible: np.ndarray   # ritmo con que los frentes activos queman cada nodo


# ----------------------------------------------------------------------------
# Dinámica del incendio
# ----------------------------------------------------------------------------
class Incendio:
    def __init__(self, bosque, params, rng_mundo):
        self.bosque = bosque
        self.params = params
        n, E = bosque.n, len(bosque.u)
        self.t = 0.0
        self.b = np.zeros(n)
        self.estado = np.full(n, SANO, dtype=int)
        self.t_frenado = np.zeros(n)        # horas seguidas con supresión >= fuego
        # El "mundo": cada arista, viva o muerta, se sortea una única vez.
        self.viva = rng_mundo.random(E) < bosque.p
        self.armada = np.zeros(E, dtype=bool)   # aristas vivas ya activadas
        self.foco = params.nodo_inicio          # el incendio empieza aquí

    # -- consultas ----------------------------------------------------------
    def _abierto(self):
        """Nodos que todavía pueden quemarse más."""
        return (self.estado == SANO) | (self.estado == ARDIENDO)

    def _ritmo(self):
        """Ritmo (fracción/hora) con que los frentes activos queman cada nodo."""
        bq = self.bosque
        abierto = self._abierto()
        alimenta = self.armada & abierto[bq.v]
        ritmo = suma_por_indice(bq.v[alimenta],
                                bq.velocidad[bq.v[alimenta]] * bq.m[alimenta], bq.n)
        if abierto[self.foco]:                  # el foco inicial es un frente más
            ritmo[self.foco] += bq.velocidad[self.foco]
        return ritmo

    def observar(self):
        return Observacion(self.t, self.b.copy(), self.estado.copy(), self._ritmo())

    def terminado(self):
        """No queda ningún frente que alimente a un nodo abierto."""
        bq = self.bosque
        abierto = self._abierto()
        alimenta = self.armada & abierto[bq.v]
        return not (alimenta.any() or abierto[self.foco])

    # -- avance -------------------------------------------------------------
    def avanzar(self, S, dt):
        """
        Avanza dt horas con una supresión S[v] sobre cada nodo.
        Orden: crecimiento -> extinción -> activación de nuevas aristas.
        """
        p, bq = self.params, self.bosque
        ritmo = self._ritmo()
        neto = np.maximum(0.0, ritmo - S)

        # 1. Crecimiento de lo quemado (nunca decrece)
        self.b = np.minimum(1.0, self.b + neto * dt)
        self.estado[(self.estado == SANO) & (self.b > 0)] = ARDIENDO
        self.estado[(self.estado == ARDIENDO) & (self.b >= 1.0 - EPS)] = QUEMADO

        # 2. Extinción: supresión >= fuego durante t_extincion horas seguidas
        atacado = (ritmo > 0) & self._abierto()
        frenado = atacado & (S >= ritmo)
        self.t_frenado[frenado] += dt
        self.t_frenado[~frenado] = 0.0
        extinguir = frenado & (self.t_frenado >= p.t_extincion - EPS)
        self.estado[extinguir] = EXTINGUIDO

        # 3. Activación: una arista viva se activa cuando su origen, en llamas,
        #    supera theta. Los nodos extinguidos ya no contagian.
        contagia = ((self.estado == ARDIENDO) | (self.estado == QUEMADO)) \
            & (self.b >= p.theta - EPS)
        self.armada |= self.viva & ~self.armada & contagia[bq.u]

        self.t += dt

    # -- resultados ---------------------------------------------------------
    def resumen(self):
        """Destrucción ponderada y desglosada (fracciones entre 0 y 1)."""
        w = self.bosque.valor(self.params)
        pers, bio = self.bosque.personas, self.bosque.biodiversidad
        return {
            "destruccion": float((w * self.b).sum() / w.sum()),
            "personas": float((pers * self.b).sum() / pers.sum()) if pers.sum() > 0 else 0.0,
            "biodiversidad": float((bio * self.b).sum() / bio.sum()) if bio.sum() > 0 else 0.0,
            "nodos_equivalentes": float(self.b.sum()),
            "duracion": self.t,
            "extinguidos": int((self.estado == EXTINGUIDO).sum()),
        }