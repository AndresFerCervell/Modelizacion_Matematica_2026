import math
import sys
import networkx as nx
import matplotlib.pyplot as plt
from core import leer_grafo
from engine import IncendioSimulator



def dibujar_evolucion(ruta_grafo, plan, q=1, guardar=None, mostrar=True, columnas=3):
    """
    Reproduce el plan de cortafuegos etapa a etapa y dibuja una imagen por etapa.
      - rojo     : zona que se acaba de incendiar en esa etapa (I_t)
      - naranja  : zona quemada en etapas anteriores
      - azul     : zona sin quemar
      - rojo discontinuo : conexión con algún cortafuegos
    plan: lista de etapas, cada una con las aristas (u, v) a cortar (como devuelve
          resolver_reto2_exacto).
    guardar: ruta del PNG a guardar (opcional).
    mostrar: si True, abre la ventana de matplotlib (se puede guardar Y mostrar).
    """
    G0 = leer_grafo(ruta_grafo, peso_unitario=True)
    G0u = G0.to_undirected()
    # Posiciones fijas para que los nodos no se muevan entre imágenes
    pos = nx.spring_layout(G0u, seed=42)
    pares = {tuple(sorted(a)) for a in G0.edges()}  # cada conexión una sola vez

    sim = IncendioSimulator(G0)
    sim.iniciar_incendio(q)

    # Cada "foto" = (título, I_t, Q_t, grafo actual)
    fotos = [(f"Inicio: fuego en {q}", set(sim.I_t), set(sim.Q_t), sim.G.copy())]
    etapa = 0
    while sim.I_t:
        cortes = plan[etapa] if etapa < len(plan) else ()
        for u, v in cortes:
            sim.bomberos_cortar_arista(u, v)
        sim.pasar_turno()
        titulo = f"Etapa {etapa + 1}: cortes {list(cortes)}\nquemadas: {len(sim.Q_t)}"
        fotos.append((titulo, set(sim.I_t), set(sim.Q_t), sim.G.copy()))
        etapa += 1

    filas = math.ceil(len(fotos) / columnas)
    fig, axs = plt.subplots(filas, columnas, figsize=(6 * columnas, 5.5 * filas))
    axs = list(axs.flat) if hasattr(axs, "flat") else [axs]

    for ax, (titulo, I_t, Q_t, G) in zip(axs, fotos):
        # Una conexión está "cortada" si falta alguno de sus dos arcos
        cortadas = [p for p in pares if not (G.has_edge(*p) and G.has_edge(p[1], p[0]))]
        cortadas_set = set(cortadas)
        intactas = [p for p in pares if p not in cortadas_set]
        colores = ['red' if n in I_t else 'orange' if n in Q_t else 'lightblue'
                   for n in G0.nodes()]
        nx.draw_networkx_edges(G0u, pos, edgelist=intactas,
                               edge_color='lightgray', ax=ax)
        nx.draw_networkx_edges(G0u, pos, edgelist=cortadas,
                               edge_color='red', style='dashed', width=2, ax=ax)
        nx.draw_networkx_nodes(G0, pos, node_color=colores, node_size=450,
                               edgecolors='black', ax=ax)
        nx.draw_networkx_labels(G0, pos, font_size=9, font_weight='bold', ax=ax)
        ax.set_title(titulo, fontsize=9)
        ax.axis('off')
    for ax in axs[len(fotos):]:
        ax.axis('off')

    plt.tight_layout()
    if guardar:
        plt.savefig(guardar, dpi=110)
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    from reto2 import resolver_reto2_exacto  # import local: evita el ciclo

    ruta = sys.argv[1] if len(sys.argv) > 1 else "data/graph_007_probs.txt"
    quemados, salvados, plan = resolver_reto2_exacto(ruta, q=1, k=2)
    print(f"Quemadas: {quemados} | Salvadas: {salvados}")
    dibujar_evolucion(ruta, plan, q=1, guardar="evolucion_graph_007.png", mostrar=True)