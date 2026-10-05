"""
Reto 5 - Efectivos de extinción.

Los dos tipos comparten un mismo primitivo: SUPRESIÓN de una cierta fuerza sobre
un nodo (fracción de nodo por hora que anulan al fuego).

  * Terrestre: presencia continua en un nodo mientras esté asignado; sin recarga;
    tiempo de preparación largo; solo actúa a como mucho `rango` saltos de su base.
  * Aéreo: una descarga de gran fuerza durante `duracion` horas y luego recarga
    (`t_recarga`) antes de poder volver a descargar; preparación corta; sin límite
    de alcance.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from reto5_modelo import EPS, _lineas_utiles

TERRESTRE, AEREO = "terrestre", "aereo"

# Estados de un efectivo
EN_PREPARACION = "en_preparacion"   # aún no está listo (cuenta atrás en `contador`)
DISPONIBLE = "disponible"           # listo para que la política lo asigne
TRABAJANDO = "trabajando"           # terrestre aplicando supresión en `nodo`
DESCARGANDO = "descargando"         # aéreo en plena descarga sobre `nodo`
EN_RECARGA = "en_recarga"           # aéreo recargando (cuenta atrás en `contador`)


@dataclass
class Efectivo:
    id: int
    tipo: str
    fuerza: float        # fracción de nodo/hora que anula al fuego
    duracion: float      # horas que dura una descarga (solo aéreos)
    t_listo: float       # horas hasta estar listo tras el inicio del incendio
    t_recarga: float     # horas entre una descarga y la siguiente (solo aéreos)
    rango: int           # saltos máximos desde la base (-1 = sin límite)
    base: int            # nodo de la estación base (-1 = ninguna)
    estado: str = EN_PREPARACION
    contador: float = 0.0
    nodo: Optional[int] = None

    def reiniciar(self):
        self.nodo = None
        if self.t_listo > EPS:
            self.estado, self.contador = EN_PREPARACION, self.t_listo
        else:
            self.estado, self.contador = DISPONIBLE, 0.0

    def copia(self):
        e = Efectivo(self.id, self.tipo, self.fuerza, self.duracion, self.t_listo,
                     self.t_recarga, self.rango, self.base)
        e.reiniciar()
        return e

    def libre_para_asignar(self):
        """La política puede decidir sobre él en este instante."""
        if self.tipo == TERRESTRE:      # un terrestre se puede reasignar cada tick
            return self.estado in (DISPONIBLE, TRABAJANDO)
        return self.estado == DISPONIBLE  # un aéreo, solo si está listo y recargado

    def puede_actuar_en(self, nodo, bosque):
        if self.tipo == AEREO or self.rango < 0:
            return True
        return bosque.saltos[self.base, nodo] <= self.rango


def leer_efectivos(ruta):
    """Formato: 'tipo fuerza duracion t_listo t_recarga rango base' (uno por línea)."""
    efectivos = []
    for i, t in enumerate(_lineas_utiles(ruta)):
        efectivos.append(Efectivo(id=i, tipo=t[0], fuerza=float(t[1]),
                                  duracion=float(t[2]), t_listo=float(t[3]),
                                  t_recarga=float(t[4]), rango=int(t[5]),
                                  base=int(t[6])))
        efectivos[-1].reiniciar()
    return efectivos


class Brigada:
    """Gestiona los efectivos durante una simulación: asignación, supresión y relojes."""

    def __init__(self, plantilla):
        self.efectivos = [e.copia() for e in plantilla]

    def asignar(self, asignacion, bosque):
        """
        asignacion: dict {id_efectivo: nodo}. Valida que cada decisión sea posible.
        Un terrestre libre sin asignar queda disponible; un aéreo sin asignar espera.
        """
        for e in self.efectivos:
            if not e.libre_para_asignar():
                continue
            nodo = asignacion.get(e.id)
            if nodo is None:
                if e.tipo == TERRESTRE:
                    e.estado, e.nodo = DISPONIBLE, None
                continue
            if not e.puede_actuar_en(nodo, bosque):
                raise ValueError(f"Efectivo {e.id} ({e.tipo}) no alcanza el nodo {nodo}")
            e.nodo = nodo
            if e.tipo == TERRESTRE:
                e.estado = TRABAJANDO
            else:
                e.estado, e.contador = DESCARGANDO, e.duracion

    def supresion(self, n, rng_ruido, sigma):
        """Supresión total S[v] aplicada en este tick, con variación aleatoria por efectivo."""
        S = np.zeros(n)
        for e in self.efectivos:
            if e.estado in (TRABAJANDO, DESCARGANDO):
                S[e.nodo] += e.fuerza * (1.0 + sigma * (2.0 * rng_ruido.random() - 1.0))
        return S

    def avanzar(self, dt):
        """Relojes: preparación, duración de la descarga y recarga."""
        for e in self.efectivos:
            if e.estado == EN_PREPARACION:
                e.contador -= dt
                if e.contador <= EPS:
                    e.estado, e.contador = DISPONIBLE, 0.0
            elif e.estado == DESCARGANDO:
                e.contador -= dt
                if e.contador <= EPS:
                    e.nodo = None
                    if e.t_recarga > EPS:
                        e.estado, e.contador = EN_RECARGA, e.t_recarga
                    else:
                        e.estado, e.contador = DISPONIBLE, 0.0
            elif e.estado == EN_RECARGA:
                e.contador -= dt
                if e.contador <= EPS:
                    e.estado, e.contador = DISPONIBLE, 0.0