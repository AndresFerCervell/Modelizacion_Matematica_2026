import networkx as nx
import matplotlib.pyplot as plt

def leer_grafo(ruta_archivo, probabilidades_inicio=None):
    """
    Lee el listado de arcos con sus probabilidades y crea un DiGraph de NetworkX.
    Formato esperado por línea: origen destino probabilidad (ej: '12 18 0.6155')
    """
    G = nx.DiGraph()
    
    # 1. Crear SIEMPRE los 20 nodos (del 0 al 19)
    # Esto asegura que el simulador no falle si intentamos incendiar 
    # un nodo que casualmente no tiene conexiones en este grafo específico.
    for i in range(20):
        prob_ini = probabilidades_inicio.get(i, 0.0) if probabilidades_inicio else 0.0
        G.add_node(i, quemado=False, prob_inicio=prob_ini)
        
    # 2. Leer los arcos del archivo
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue # Ignorar líneas vacías
            
            partes = linea.split()
            if len(partes) >= 3:
                try:
                    origen = int(partes[0])
                    destino = int(partes[1])
                    peso = float(partes[2].replace(',', '.')) # Soporta comas o puntos
                    
                    if peso > 0:
                        G.add_edge(origen, destino, weight=peso)
                except ValueError:
                    print(f"Advertencia: No se pudo procesar la línea -> {linea}")

    return G

def dibujar_grafo(G):
    """
    Dibuja el grafo usando Matplotlib. Colorea en rojo los nodos quemados.
    Muestra los pesos (probabilidades) en las aristas.
    """
    plt.figure(figsize=(10, 8))
    
    # Posicionamiento de los nodos
    pos = nx.spring_layout(G, seed=42) # Semilla fija 
    
    # Colores: Rojo si está quemado, Azul claro si no
    colores_nodos = ['red' if G.nodes[n].get('quemado', False) else 'lightblue' for n in G.nodes()]
    
    # Dibujar nodos y etiquetas
    nx.draw_networkx_nodes(G, pos, node_color=colores_nodos, node_size=500, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_weight='bold')
    
    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
    
    # Etiquetas de aristas (probabilidades)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)
    
    plt.title("Estado del Bosque (Reto Lo Que Arde)")
    plt.axis('off')
    plt.show()