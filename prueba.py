from core import leer_grafo, dibujar_grafo
from engine import IncendioSimulator

# 1. Leer datos
grafo_base = leer_grafo("data/graph_007_probs.txt")

# 2. Inicializar simulador (con semilla para reproducibilidad)
sim = IncendioSimulator(grafo_base, semilla=12345)

# 3. Iniciar fuego (Reto 2 y 3 indican q=1)
sim.iniciar_incendio(nodo_inicial=1)
print("Estado inicial:", sim.estado_actual())

# 4. Turno 1: Los bomberos actúan (k=1)
sim.bomberos_cortar_arista(1, 9) # Ejemplo: cortan la arista 1 -> 9

# 5. Turno 1: El fuego avanza
sim.pasar_turno()
print("Tras etapa 1:", sim.estado_actual())

dibujar_grafo(sim.G)