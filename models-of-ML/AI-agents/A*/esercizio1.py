class Node:

    def __init__(self, state, h, path_cost=0, parent=None):
        self.state = state
        self.h = h
        self.path_cost = path_cost
        self.parent = parent

    def to_solution(self):
        seq = []
        node = self
        s0 = None
        while node is not None:
            if node.parent is None:
                s0 = node.state
            if node.parent is not None:
                seq.append(node.state)
            node = node.parent
        assert s0 is not None
        return list(reversed(seq))
    
    def __repr__(self):
        s = f'Node(state={self.state}, path_cost={self.path_cost}'
        s += ')' if self.parent is None else f', parent={self.parent.state})'
        return s

def a_star(initial_state, goal_test, successor_fn, heuristic_fn):
  
    # calcoliamo la distanza dal goal dall'inizio
    h = heuristic_fn(initial_state)
    g, f = 0, h

    to_be_explored, explored_set = [], set() # il primo sono i nodi ancora da esplorare
                                             # il secondo sono i nodi già esplorati nella ricerca

    # il primo nodo da esplorare è quello da cui l'agente parte
    root = Node(initial_state, h)

    to_be_explored.append((f, root))
    # finché abbiamo nodi da esplorare
    while len(to_be_explored) != 0:
        
        # ordino in base al costo crescente
        to_be_explored.sort(key=lambda x: x[0])


        # prendo il nodo con costo minore
        _, current_node = to_be_explored.pop(0)

        # se siamo arrivati al goal basta ritornare il nodo
        # la funzione to_solution si occuperà di riprendersi 
        # il percorso tramite i parent dei nodi
        if goal_test(current_node.state):
            return current_node

        # se siamo qua non siamo arrivati al goal
        # tocca andare avanti,
        # aggiungiamo il nodo corrente nei nodi esplorati
        explored_set.add(current_node.state)

        for successore, costo in successor_fn(current_node.state):
            # se il successore già è negli esplorati (insieme chiuso)
            # non si va in quella direzione
            if successore in explored_set:
                continue

            # se il successore è da esplorare ci calcoliamo f
            gsucc = current_node.path_cost + costo

            fsucc = gsucc + heuristic_fn(successore)

            nodesucc = Node(successore, heuristic_fn(successore), path_cost=gsucc, parent=current_node)

            # esiste un percorso migliore grazie al nodo corrente?
            better_path_bool = any(node.state == successore and gsucc < node.path_cost for f, node in to_be_explored)

            # se non esiste aggiungiamo semplicemente il nodo alla lista dei nodi da esplorare
            if better_path_bool == False:
                to_be_explored.append((fsucc, nodesucc))
            # registriamo il percorso migliore
            else:
                to_be_explored = [(fsucc, nodesucc) if node.state == successore and gsucc < node.path_cost else (f, node) for f, node in to_be_explored]

    return None


# Commento per motivare la scelta dell'euristica:
# Ho utilizzato la distanza di Manhattan poiché la struttura a griglia e la possibilità di muoversi solo nelle
# 4 direzioni (sopra, sotto, sinistra, destra) lo consigliavano. 

# Altre euristiche, come la distanza euclidea contemplano nel calcolo anche la possibilità di andare in diagonale,
# mentre la distanza di Manhattan è perfetta per scenari in cui ci si muove solo in orizzontale o verticale.
def heuristic(s, goal, is_solid):
    h_s = abs(s[0] - goal[0]) + abs(s[1] - goal[1]) 
    return h_s


