import itertools
from core import leer_grafo
from engine import IncendioSimulator
from evolucion import dibujar_evolucion


def alcance_por_capas(G, I_t, excluida=None, H=3):
    """
    BFS por capas desde I_t ignorando el arco 'excluida'.
    Devuelve (nodos alcanzados en como mucho H etapas, nodos alcanzados en total).
    """
    vistos, capa, en_h = set(I_t), set(I_t), 0
    d = 0
    while capa:
        d += 1
        sig = set()
        for x in capa:
            for y in G.successors(x):
                if (x, y) == excluida or y in vistos or G.nodes[y]['quemado']:
                    continue
                sig.add(y)
        vistos |= sig
        capa = sig
        if d <= H:
            en_h = len(vistos) - len(I_t)
    return en_h, len(vistos) - len(I_t)

# =====================================================================
# 1. MÉTODO EXACTO: BRANCH & BOUND
# =====================================================================
# Idea clave (p_ij = 1):
#  - Cortar (u, v) solo sirve si se corta ANTES de que u propague, es decir,
#    como muy tarde en la etapa t en la que u arde ("fecha límite" t).
#  - Un nodo v de la frontera solo se salva en la etapa t si se cortan TODOS
#    los arcos que le llegan desde I_t (cuesta c_v cortes). Cortar solo
#    alguno es inútil: v arde igualmente.
#  - Capacidad: los cortes con fecha límite <= t caben si y solo si su total
#    acumulado es <= k*(t+1). Los cortes sobrantes de etapas anteriores se
#    pueden usar para "adelantar" cortes. Por tanto el estado solo necesita
#    el crédito acumulado, y no hace falta enumerar combinaciones de aristas
#    de todo el grafo: en cada etapa se ramifica solo sobre qué nodos de la
#    frontera se protegen.

def cota_optimista_quemados(I_t, Q_t, G, credito):
    """
    Cota inferior VÁLIDA del nº final de quemados: cada nodo protegido cuesta
    al menos 1 corte, así que de los |N| vecinos amenazados se salvan como
    mucho 'credito' y los demás arderán.
    """
    vecinos = {v for u in I_t for v in G.successors(u) if not G.nodes[v]['quemado']}
    return len(Q_t) + max(0, len(vecinos) - credito)


def branch_and_bound(sim, credito, decisiones, k, mejor_resultado, memo):
    """
    Búsqueda recursiva. Antes eran variables de la función exterior; ahora se
    pasan como argumentos:
      k                : cortes nuevos por etapa
      mejor_resultado  : dict mutable {"min_quemados", "decisiones"} (se actualiza in situ)
      memo             : set mutable de estados ya explorados (se actualiza in situ)
    """
    # 1. Fuego apagado: evaluamos
    if len(sim.I_t) == 0:
        if len(sim.Q_t) < mejor_resultado["min_quemados"]:
            mejor_resultado["min_quemados"] = len(sim.Q_t)
            mejor_resultado["decisiones"] = list(decisiones)
        return

    # 2. PODA POR COTA
    if cota_optimista_quemados(sim.I_t, sim.Q_t, sim.G, credito) >= mejor_resultado["min_quemados"]:
        return

    # 3. MEMORIZACIÓN
    estado_key = (frozenset(sim.I_t), frozenset(sim.Q_t), credito)
    if estado_key in memo:
        return
    memo.add(estado_key)

    # 4. Nodos amenazados y coste de protegerlos (arcos que les llegan desde I_t)
    coste = {}
    for u in sim.I_t:
        for v in sim.G.successors(u):
            if not sim.G.nodes[v]['quemado']:
                coste[v] = coste.get(v, 0) + 1
    amenazados = list(coste)

    # 5. Ramificar sobre qué amenazados se protegen (cualquier subconjunto que quepa)
    for r in range(len(amenazados) + 1):
        for protegidos in itertools.combinations(amenazados, r):
            gasto = sum(coste[v] for v in protegidos)
            if gasto > credito:
                continue
            cortes = [(u, v) for v in protegidos for u in sim.I_t if sim.G.has_edge(u, v)]

            sim_rama = IncendioSimulator(sim.G)
            sim_rama.I_t = set(sim.I_t)
            sim_rama.Q_t = set(sim.Q_t)
            sim_rama.t = sim.t
            for u, v in cortes:
                sim_rama.bomberos_cortar_arista(u, v)
            sim_rama.pasar_turno()

            # El crédito de la etapa siguiente suma k nuevos cortes
            branch_and_bound(sim_rama, credito - gasto + k, decisiones + [(sim.t, cortes)],
                             k, mejor_resultado, memo)


def resolver_reto2_exacto(ruta_grafo, q=1, k=2):
    grafo_base = leer_grafo(ruta_grafo, peso_unitario=True)  # p_ij = 1
    n_nodos = grafo_base.number_of_nodes()

    sim_inicial = IncendioSimulator(grafo_base)
    sim_inicial.iniciar_incendio(q)

    mejor_resultado = {"min_quemados": n_nodos + 1, "decisiones": []}
    memo = set()  # estados ya explorados: (I_t, Q_t, crédito) determina el futuro

    branch_and_bound(sim_inicial, k, [], k, mejor_resultado, memo)

    zonas_salvadas = n_nodos - mejor_resultado["min_quemados"]
    plan = planificar_cortes(mejor_resultado["decisiones"], k)
    return mejor_resultado["min_quemados"], zonas_salvadas, plan


def planificar_cortes(decisiones, k):
    """
    Convierte las decisiones (fecha_límite, [aristas]) en un calendario real
    de como mucho k cortes por etapa (primero los de fecha límite más cercana).
    """
    pendientes = sorted((t, a) for t, cortes in decisiones for a in cortes)
    return [tuple(a for _, a in pendientes[i:i + k]) for i in range(0, len(pendientes), k)]

# =====================================================================
# 2. MÉTODO APROXIMADO: GREEDY CON HORIZONTE (para grafos grandes)
# =====================================================================

def elegir_cortafuegos_reto2_greedy(G, I_t, Q_t, H=3):
    """
    Corta el arco de la frontera que minimiza las zonas alcanzadas en H etapas
    (desempate: alcance total). Mirar solo el alcance total no sirve: en grafos
    bien conectados casi ningún corte aislado lo cambia y todo empata.
    """
    mejor, mejor_val = None, None
    for u in I_t:
        for v in G.successors(u):
            if not G.nodes[v]['quemado']:
                val = alcance_por_capas(G, I_t, (u, v), H)
                if mejor_val is None or val < mejor_val:
                    mejor, mejor_val = (u, v), val
    return mejor

def resolver_reto2_aproximado(ruta_grafo, q=1, k=2, horizontes=(1, 2, 3)):
    """Ejecuta el greedy con varios horizontes (determinista) y se queda con el mejor."""
    grafo_base = leer_grafo(ruta_grafo, peso_unitario=True)
    mejor = grafo_base.number_of_nodes()
    
    for H in horizontes:
        sim = IncendioSimulator(grafo_base)
        heur = lambda G, I_t, Q_t, H=H: elegir_cortafuegos_reto2_greedy(G, I_t, Q_t, H)
        mejor = min(mejor, sim.simular_episodio(nodo_inicio=q, k=k, heuristica=heur))
    return mejor, grafo_base.number_of_nodes() - mejor

def probar_aproximado(g, q=1, k=2):
    try:
        q_ap, s_ap = resolver_reto2_aproximado(g, q=q, k=k)
        print(f"[{g}] -> Quemados: {q_ap:2d} | Salvados: {s_ap:2d}")
    except FileNotFoundError:
        print(f"[{g}] no encontrado")

# =====================================================================
# EJECUCIÓN
# =====================================================================

if __name__ == "__main__":

    print("1. SOLUCIÓN EXACTA (Branch & Bound) - graph_007, q=1, k=2")

    quemados, salvados, plan = resolver_reto2_exacto("data/graph_007_probs.txt", q=1, k=2)

    print(f"Zonas quemadas: {quemados} | ZONAS SALVADAS: {salvados}")
    for t, cortes in enumerate(plan):
        print(f"  Etapa {t}: cortar {cortes}")

    dibujar_evolucion(ruta_007, plan, q=1, guardar="evolucion_graph_007.png", mostrar=True)

    print("\n2. SOLUCIÓN APROXIMADA (Greedy)")
       
    probar_aproximado("data/graph_003_probs.txt")
    probar_aproximado("data/graph_004_probs.txt")
    probar_aproximado("data/graph_005_probs.txt")
    probar_aproximado("data/graph_007_probs.txt")
    probar_aproximado("data/graph_008_probs.txt")
    probar_aproximado("data/graph_010_probs.txt")