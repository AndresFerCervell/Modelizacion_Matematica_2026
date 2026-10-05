import random
from core import leer_grafo
from engine import IncendioSimulator
from reto3 import elegir_cortafuegos_reto3, resolver_reto3

def resolver_reto4_piromano(ruta_grafo, k=1, num_simulaciones=1000):
    """
    Evalúa todos los nodos iniciales posibles para encontrar el peor escenario (máximo daño esperado),
    asumiendo que los bomberos se defienden con la heurística Greedy.
    """
    grafo_base = leer_grafo(ruta_grafo)
    
    # Semillas fijas 
    random.seed(42)
    semillas_simulacion = [random.randint(0, 999999) for _ in range(num_simulaciones)]
    
    peor_nodo = -1
    max_danio_esperado = -1.0
    
    print(f"Analizando {ruta_grafo}...")
    
    # Evaluamos exhaustivamente cada uno de los 20 nodos 
    for q in grafo_base.nodes():
        quemados_con_bomberos = 0
        
        for semilla in semillas_simulacion:
            sim = IncendioSimulator(grafo_base, semilla=semilla)
            # El fuego empieza en 'q', los bomberos se defienden cortando 'k' aristas
            quemados_con_bomberos += sim.simular_episodio(
                nodo_inicio=q, 
                k=k, 
                heuristica=elegir_cortafuegos_reto3
            )
            
        danio_medio_q = quemados_con_bomberos / num_simulaciones
        # print(f"  Nodo {q:02d} -> Daño esperado: {danio_medio_q:.2f} zonas") 
        
        if danio_medio_q > max_danio_esperado:
            max_danio_esperado = danio_medio_q
            peor_nodo = q
            
    print(f">>> PEOR ESCENARIO: Fuego en zona {peor_nodo} (Quema {max_danio_esperado:.2f} zonas de media)\n")
    return peor_nodo, max_danio_esperado

if __name__ == "__main__":
    grafos = [
        "data/graph_003_probs.txt", "data/graph_004_probs.txt", 
        "data/graph_005_probs.txt", "data/graph_007_probs.txt", 
        "data/graph_008_probs.txt", "data/graph_010_probs.txt"
    ]
    for g in grafos:
        resolver_reto4_piromano(g, k=1, num_simulaciones=1000)
        