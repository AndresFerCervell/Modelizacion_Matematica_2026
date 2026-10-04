import random
from core import leer_grafo
from engine import IncendioSimulator

def elegir_cortafuegos_reto3(G, I_t, Q_t):
    """
    Heurística Greedy para el Reto 3.
    Evalúa la Esperanza Matemática del Daño a 1 paso.
    """
    mejor_arista = None
    max_score = -1.0
    
    # Analizamos solo la "frontera" del incendio
    for u in I_t:
        for v in G.successors(u):
            # Si el vecino v aún no está quemado, es un candidato a proteger
            if not G.nodes[v]['quemado']:
                prob_cruce = G.edges[u, v]['weight']
                
                # Calculamos el daño esperado si el fuego llega a v
                daño_esperado_v = 0.0
                for w in G.successors(v):
                    # No sumamos nodos ya quemados ni volvemos hacia atrás
                    if not G.nodes[w]['quemado'] and w != u:
                        daño_esperado_v += G.edges[v, w]['weight']
                
                # Score = probabilidad de que salte * (1 zona de v + daño colateral que causaría)
                score = prob_cruce * (1.0 + daño_esperado_v) 
                
                if score > max_score:
                    max_score = score
                    mejor_arista = (u, v)
                    
    return mejor_arista

def resolver_reto3(ruta_grafo, q=1, k=1, num_simulaciones=1000):
    """
    Ejecuta el experimento Montecarlo para calcular las zonas salvadas.
    """
    # Fijamos la semilla para las condiciones experimentales
    random.seed(42)
    semillas_simulacion = [random.randint(0, 999999) for _ in range(num_simulaciones)]
    
    grafo_base = leer_grafo(ruta_grafo)
    
    # FASE 1: Fuego sin intervención de los bomberos
    quemados_sin_bomberos = 0
    for semilla in semillas_simulacion:
        
        sim = IncendioSimulator(grafo_base, semilla=semilla)
        quemados_sin_bomberos += sim.simular_episodio(nodo_inicio=q, k=0, heuristica=None)
        
    media_sin_bomberos = quemados_sin_bomberos / num_simulaciones
    
    # FASE 2: Intervención Greedy 
    quemados_con_bomberos = 0
    for semilla in semillas_simulacion:
        sim = IncendioSimulator(grafo_base, semilla=semilla)
        quemados_con_bomberos += sim.simular_episodio(nodo_inicio=q, k=k, heuristica=elegir_cortafuegos_reto3)
        
    media_con_bomberos = quemados_con_bomberos / num_simulaciones
    
    # RESULTADOS
    zonas_salvadas = media_sin_bomberos - media_con_bomberos
    
    print(f"--- Resultados para {ruta_grafo} ---")
    print(f"Quemados medios (Sin bomberos): {media_sin_bomberos:.2f} zonas")
    print(f"Quemados medios (Con Greedy)  : {media_con_bomberos:.2f} zonas")
    print(f"ZONAS SALVADAS ESPERADAS      : {zonas_salvadas:.2f} zonas\n")

# Ejecución para probar 
if __name__ == "__main__":
    resolver_reto3("data/graph_003_probs.txt", q=1, k=1, num_simulaciones=1000)
    resolver_reto3("data/graph_004_probs.txt", q=1, k=1, num_simulaciones=1000)
    resolver_reto3("data/graph_005_probs.txt", q=1, k=1, num_simulaciones=1000)
    resolver_reto3("data/graph_007_probs.txt", q=1, k=1, num_simulaciones=1000)
    resolver_reto3("data/graph_008_probs.txt", q=1, k=1, num_simulaciones=1000)
    resolver_reto3("data/graph_010_probs.txt", q=1, k=1, num_simulaciones=1000)
    