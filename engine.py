import random
import networkx as nx

class IncendioSimulator:
    """
    Motor que gestiona la evolución del incendio por etapas discretas.
    """
    def __init__(self, grafo, semilla=None):
        """
        Inicializa el simulador con un grafo.
        
        Args:
            grafo (nx.DiGraph): El grafo del bosque (modificable).
            semilla (int, opcional): Semilla para la aleatoriedad, garantizando reproducibilidad.
        """
        self.G = grafo.copy() # Trabajamos sobre una copia para no alterar el original
        self.t = 0 # Etapa actual
        self.I_t = set() # Zonas incendiadas JUSTO en la etapa t
        self.Q_t = set() # Zonas totalmente quemadas (histórico)
        
        if semilla is not None:
            random.seed(semilla)
            
    def iniciar_incendio(self, nodo_inicial):
        """
        Fuerza el inicio del incendio en un nodo específico (Etapa 0).
        """
        if self.G.has_node(nodo_inicial):
            self.G.nodes[nodo_inicial]['quemado'] = True
            self.I_t.add(nodo_inicial)
            self.Q_t.add(nodo_inicial)
            self.t = 0
            return True
        return False

    def pasar_turno(self):
        """
        Avanza la simulación a la etapa t+1.
        Las zonas en I_t intentan propagar el fuego a sus vecinos no quemados
        basándose en la probabilidad p_ij de la arista.
        
        Returns:
            bool: True si el fuego sigue activo (hay nuevos nodos quemados), False si se extinguió.
        """
        self.t += 1
        nuevos_incendiados = set()
        
        # Para cada nodo que se acaba de incendiar en la etapa anterior...
        for u in self.I_t:
            # Miramos sus vecinos hacia los que hay un arco
            for v in self.G.successors(u):
                # Si el vecino no está quemado todavía
                if not self.G.nodes[v]['quemado']:
                    # Tiramos los dados basados en el peso (probabilidad p_ij)
                    probabilidad = self.G.edges[u, v]['weight']
                    if random.random() <= probabilidad:
                        nuevos_incendiados.add(v)
        
        # Actualizamos el estado de los nodos en el grafo
        for v in nuevos_incendiados:
            self.G.nodes[v]['quemado'] = True
            
        # Actualizamos los conjuntos matemáticos
        self.I_t = nuevos_incendiados
        self.Q_t.update(nuevos_incendiados)
        
        # El fuego sigue si I_t no está vacío
        return len(self.I_t) > 0

    def bomberos_cortar_arista(self, u, v):
        """
        Acción de los bomberos: elimina permanentemente la arista dirigida (u, v)
        para crear un cortafuegos y evitar la propagación.
        
        Args:
            u (int): Nodo origen.
            v (int): Nodo destino.
            
        Returns:
            bool: True si la arista existía y fue eliminada, False en caso contrario.
        """
        if self.G.has_edge(u, v):
            self.G.remove_edge(u, v)
            # Nota para la memoria: Como el modelo indica que (i,j) y (j,i) pueden tener p_ij distintas,
            # los bomberos aquí cortan una arista dirigida. Si un cortafuegos real corta ambos sentidos,
            # deberíais llamar a este método también con (v, u).
            return True
        return False
        
    def estado_actual(self):
        """Devuelve un resumen del estado de la simulación."""
        return {
            "etapa": self.t,
            "fuego_activo": len(self.I_t) > 0,
            "incendiados_ahora": list(self.I_t),
            "total_quemados": len(self.Q_t),
            "zonas_salvadas": self.G.number_of_nodes() - len(self.Q_t)
        }