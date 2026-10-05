"""
Reto 5 - Políticas de asignación de efectivos.

Una política decide, en cada tick, a qué nodo va cada efectivo libre. Solo ve la
Observacion (estado visible del fuego) y los datos conocidos del bosque; nunca el
sorteo de aristas vivas/muertas.

    decidir(obs, bosque, efectivos, params) -> {id_efectivo: nodo}
"""
from dataclasses import dataclass

import numpy as np

from reto5_efectivos import AEREO, DESCARGANDO
from reto5_modelo import ARDIENDO, SANO, suma_por_indice

EPS = 1e-12


# ----------------------------------------------------------------------------
# Riesgo de cada nodo
# ----------------------------------------------------------------------------
@dataclass
class Riesgo:
    urgencia: np.ndarray        # prob. de que el nodo esté (o vaya a estar) bajo fuego
    valor_expuesto: np.ndarray  # valor que aún está en juego (propio + aguas abajo)
    riesgo: np.ndarray          # urgencia * valor_expuesto
    necesidad: np.ndarray       # supresión (fracción/hora) para neutralizar el fuego esperado


def calcular_riesgo(obs, bosque, params):
    """
    riesgo_v = urgencia_v * valor_expuesto_v.

    urgencia: 1 si ya entra fuego por algún frente activo; si no, probabilidad de
      que le llegue pronto desde un vecino en llamas que aún no ha contagiado:
      1 - prod(1 - p_uv * b_u/theta). (Un vecino que ya superó theta sin haber
      prendido a v indica que esa arista está muerta, y no cuenta.)
    valor_expuesto: lo que queda sin quemar del nodo, w_v (1 - b_v), más una
      fracción gamma de lo que protege aguas abajo (solo si todavía no ha
      activado sus aristas: si ya contagió, apagarlo no salva a sus vecinos).
    necesidad: ritmo de los frentes visibles + ritmo esperado de los potenciales.
    """
    n = bosque.n
    w = bosque.valor(params)
    abierto = (obs.estado == SANO) | (obs.estado == ARDIENDO)
    restante = w * (1.0 - obs.b) * abierto

    # Vecinos en llamas que todavía no han superado theta (pueden contagiar pronto)
    previo = (obs.estado == ARDIENDO) & (obs.b < params.theta)
    proximidad = np.where(previo, obs.b / params.theta, 0.0)
    p_llegada = bosque.p * proximidad[bosque.u]
    log_no_llega = suma_por_indice(bosque.v,
                                   np.log1p(-np.clip(p_llegada, 0.0, 1 - 1e-12)), n)
    urg_potencial = 1.0 - np.exp(log_no_llega)
    urgencia = np.where(obs.ritmo_visible > 0, 1.0, urg_potencial) * abierto

    ritmo_potencial = suma_por_indice(
        bosque.v,
        np.where(previo[bosque.u], bosque.p * bosque.m * bosque.velocidad[bosque.v], 0.0),
        n)
    necesidad = (obs.ritmo_visible + ritmo_potencial) * abierto

    puede_contagiar = abierto & (obs.b < params.theta)
    aguas_abajo = suma_por_indice(bosque.u, bosque.p * restante[bosque.v], n) * puede_contagiar
    valor_expuesto = restante + params.gamma * aguas_abajo

    return Riesgo(urgencia, valor_expuesto, urgencia * valor_expuesto, necesidad)


# ----------------------------------------------------------------------------
# Políticas
# ----------------------------------------------------------------------------
class Politica:
    nombre = "política"

    def decidir(self, obs, bosque, efectivos, params):
        raise NotImplementedError


class PoliticaNula(Politica):
    """Referencia: los bomberos no hacen nada."""
    nombre = "Sin efectivos"

    def decidir(self, obs, bosque, efectivos, params):
        return {}


class PoliticaAleatoria(Politica):
    """Referencia: cada efectivo libre va a un nodo posible elegido al azar."""
    nombre = "Aleatoria"

    def __init__(self, semilla=0):
        self.rng = np.random.default_rng(semilla)

    def decidir(self, obs, bosque, efectivos, params):
        asignacion = {}
        for e in efectivos:
            if not e.libre_para_asignar():
                continue
            candidatos = [v for v in range(bosque.n)
                          if e.puede_actuar_en(v, bosque)
                          and obs.estado[v] in (SANO, ARDIENDO)
                          and (e.tipo != AEREO or obs.ritmo_visible[v] > 0)]
            if candidatos:
                asignacion[e.id] = int(self.rng.choice(candidatos))
        return asignacion


class PoliticaGreedyRiesgo(Politica):
    """
    Greedy por ganancia marginal. Repite hasta que no queden efectivos o
    ganancias: entre todos los pares (efectivo libre, nodo) posibles, elige el de
    mayor ganancia, lo fija y descuenta del nodo la fuerza ya aplicada.

        ganancia(e, v) = riesgo_v * min(fuerza_e, necesidad_pendiente_v) / necesidad_v

    es decir, el riesgo del nodo multiplicado por la fracción del fuego esperado
    que ese efectivo todavía puede neutralizar. Así un nodo recibe efectivos
    hasta quedar cubierto y los demás se reparten entre otros nodos.
    Restricciones: los terrestres solo alcanzan los nodos dentro de su rango, y
    los aéreos solo descargan sobre fuego visible.
    """
    nombre = "Greedy por riesgo"

    def decidir(self, obs, bosque, efectivos, params):
        r = calcular_riesgo(obs, bosque, params)
        pendiente = r.necesidad.copy()
        # Los aéreos a media descarga no se pueden redirigir, pero ya cubren su nodo
        for e in efectivos:
            if e.estado == DESCARGANDO:
                pendiente[e.nodo] -= e.fuerza
        candidatos = [v for v in range(bosque.n)
                      if r.riesgo[v] > EPS and r.necesidad[v] > EPS]
        libres = [e for e in efectivos if e.libre_para_asignar()]
        asignacion = {}

        while libres and candidatos:
            mejor_g, mejor_e, mejor_v = EPS, None, None
            for e in libres:
                for v in candidatos:
                    if pendiente[v] <= EPS or not e.puede_actuar_en(v, bosque):
                        continue
                    if e.tipo == AEREO and obs.ritmo_visible[v] <= 0:
                        continue
                    g = r.riesgo[v] * min(e.fuerza, pendiente[v]) / r.necesidad[v]
                    if g > mejor_g:
                        mejor_g, mejor_e, mejor_v = g, e, v
            if mejor_e is None:
                break
            asignacion[mejor_e.id] = mejor_v
            pendiente[mejor_v] -= mejor_e.fuerza
            libres.remove(mejor_e)
        return asignacion


class PoliticaAsignacionOptima(Politica):
    """
    PLANTEADO, SIN IMPLEMENTAR. Alternativa al greedy: resolver cada tick el
    reparto como un problema de asignación óptima (algoritmo húngaro).

    Plan:
      1. Matriz de ganancias G[e, v] = riesgo_v * min(fuerza_e, necesidad_v) / necesidad_v
         para cada efectivo libre e y nodo candidato v; las parejas imposibles
         (terrestre fuera de rango, aéreo sobre un nodo sin fuego visible) se
         ponen a 0 o a -inf.
      2. Para admitir varios efectivos en un mismo nodo, replicar la columna de
         cada nodo tantas veces como efectivos libres haya, con ganancias
         decrecientes (cada copia solo cubre la necesidad que dejan las anteriores).
      3. scipy.optimize.linear_sum_assignment(G, maximize=True) devuelve el reparto
         de ganancia total máxima; descartar las parejas de ganancia 0.

    Diferencia con el greedy: el greedy fija cada pareja de forma definitiva y
    puede gastar un efectivo flexible (aéreo) donde uno limitado (terrestre
    con rango corto) habría bastado. El húngaro mira todas las parejas a la vez.
    Sigue optimizando solo el instante actual, sin mirar el futuro.
    """
    nombre = "Asignación óptima (pendiente)"

    def decidir(self, obs, bosque, efectivos, params):
        raise NotImplementedError("Política de asignación óptima aún no implementada.")