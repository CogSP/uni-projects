

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


# nodi esplorati almeno una volta  
all_nodes = {}
# var globali usate da successors_ignore_solid()
# nel caso in cui il percorso sia irraggiungibile
# vengono correttamente assegnate in successors()
region_width_global = None
region_height_global = None

def a_star(initial_state, goal_test, successor_fn, heuristic_fn):

    all_nodes.clear()

    root = Node(initial_state, heuristic_fn(initial_state), path_cost = 0, parent = None)

    # nodi scoperti da (ri-)espandere
    openSet = []
    openSet.append(root)

    all_nodes[root.state] = root


    while len(openSet) > 0:

        # TODO: se ci sono diversi migliori nodi?
        # troviamo il nodo che ha il costo f minore
        best_f = float('inf')
        current = None
        for nodo in openSet:
            if nodo.path_cost + heuristic_fn(nodo.state) < best_f:
                best_f = nodo.path_cost + heuristic_fn(nodo.state)
                current = nodo           
            
        openSet.remove(current)


        if (goal_test(current.state) == True):
            # siamo arrivati al goal,
            # ritorniamo il nodo corrente: verrà poi recuperato tutto il percorso 
            # grazie all'attributo parent di Node
            return current
        
        # se non siamo arrivati al goal troviamo i successori di current
        successori = successor_fn(current.state)
        for successore in successori:
            
            # se il nodo successore non è mai stato visitato
            if successore.state not in all_nodes:
                all_nodes[successore.state] = successore
                successore.h = heuristic_fn(successore.state)
                successore.path_cost = float('inf')

            # vediamo se abbiamo trovato un path <root, successore>
            # meno costoso di quello attuale
            tentative_gScore = current.path_cost + distance(current.state, successore.state)
            if (tentative_gScore < successore.path_cost):
                successore.parent = current
                successore.path_cost = tentative_gScore
                
                if successore not in openSet:
                    openSet.append(successore)

    #se si è qui vuol dire che non esiste un percorso per arrivare all'obiettivo
    # l'esercizio 2 ci chiede comunque di ritornare un percorso: ripetiamo l'algoritmo 
    # ma stavolta ignorando se le celle sono solid
    
    return a_star_ignore_solid(
            initial_state, 
            goal_test, 
            lambda s: successors_ignore_solid(s), # non passo is_solid ma neanche le dimensioni: uso le var globali
            heuristic_fn
            )


    
            

        

def distance(p1, p2):
    # distanza di Manhattan
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) 




def successors(s, is_solid, region_width, region_height):

    global region_width_global
    global region_height_global

    # ne approfitto per salvarmi "region_width" e "region_height"
    # come variabili globali poiché nel caso in cui
    # il goal sia irraggiungibile ci servirà passarlo alla funzione
    # successors_ignore_solid
    region_width_global = region_width
    region_height_global = region_height

    # per ogni direzione andiamo a verificare se non si esce dai bordi 

    succ = []

    left = (s[0] - 1, s[1])
    if (is_solid(left) == False and left[0] >= 0 and left[0] < region_width and left[1] >= 0 and left[1] < region_height):
        
        if left in all_nodes:
            left_node = all_nodes[left]

        else: 
            left_node = Node(left, 0)
        

        succ.append(left_node)


    right = (s[0] + 1, s[1])
    if (is_solid(right) == False and right[0] >= 0 and right[0] < region_width and right[1] >= 0 and right[1] < region_height):
            
            if right in all_nodes:
                right_node = all_nodes[right]

            else: 
                # h = 0 è un placeholder tanto appena si ritorna alla funzione precedente viene sovrascritto
                right_node = Node(right, 0)
            

            succ.append(right_node)


    up = (s[0], s[1] - 1)
    if (is_solid(up) == False and up[0] >= 0 and up[0] < region_width and up[1] >= 0 and up[1] < region_height):
        
        if up in all_nodes:
            up_node = all_nodes[up]

        else: 
            up_node = Node(up, 0)
        

        succ.append(up_node)

    down = (s[0], s[1] + 1)
    if (is_solid(down) == False and down[0] >= 0 and down[0] < region_width and down[1] >= 0 and down[1] < region_height):
            
            if down in all_nodes:
                down_node = all_nodes[down]

            else: 
                down_node = Node(down, 0)
            

            succ.append(down_node)

    return succ
    

def heuristic(s, goal, is_solid):
    # distanza di Manhattan dal goal
    h_s = abs(s[0] - goal[0]) + abs(s[1] - goal[1]) 
    return h_s




###################################
# nel caso in cui non esista un percorso senza ignorare le celle piene

def a_star_ignore_solid(initial_state, goal_test, successor_fn, heuristic_fn):

    all_nodes.clear()

    root = Node(initial_state, heuristic_fn(initial_state), path_cost = 0, parent = None)

    # nodi scoperti da (ri-)espandere
    openSet = []
    openSet.append(root)

    all_nodes[root.state] = root


    while len(openSet) > 0:

        # TODO: se ci sono diversi migliori nodi?
        # troviamo il nodo che ha il costo f minore
        best_f = float('inf')
        current = None
        for nodo in openSet:
            if nodo.path_cost + heuristic_fn(nodo.state) < best_f:
                best_f = nodo.path_cost + heuristic_fn(nodo.state)
                current = nodo           
            
        openSet.remove(current)


        if (goal_test(current.state) == True):
            # siamo arrivati al goal,
            # ritorniamo il nodo corrente: verrà poi recuperato tutto il percorso 
            # grazie all'attributo parent di Node
            return current
        
        # se non siamo arrivati al goal troviamo i successori di current
        successori = successor_fn(current.state)
        for successore in successori:
            
            # se il nodo successore non è mai stato visitato
            if successore.state not in all_nodes:
                all_nodes[successore.state] = successore
                successore.h = heuristic_fn(successore.state)
                successore.path_cost = float('inf')

            # vediamo se abbiamo trovato un path <root, successore>
            # meno costoso di quello attuale
            tentative_gScore = current.path_cost + distance(current.state, successore.state)
            if (tentative_gScore < successore.path_cost):
                successore.parent = current
                successore.path_cost = tentative_gScore
                
                if successore not in openSet:
                    openSet.append(successore)



def successors_ignore_solid(s):

    # per ogni direzione andiamo a verificare se non si esce dai bordi 

    succ = []
    left = (s[0] - 1, s[1])
    if (left[0] >= 0 and left[0] < region_width_global and left[1] >= 0 and left[1] < region_height_global):
        
        if left in all_nodes:
            left_node = all_nodes[left]

        else: 
            left_node = Node(left, 0)
        

        succ.append(left_node)


    right = (s[0] + 1, s[1])
    if (right[0] >= 0 and right[0] < region_width_global and right[1] >= 0 and right[1] < region_height_global):
            
            if right in all_nodes:
                right_node = all_nodes[right]

            else: 
                # h = 0 è un placeholder tanto appena si ritorna alla funzione precedente viene sovrascritto
                right_node = Node(right, 0)
            

            succ.append(right_node)


    up = (s[0], s[1] - 1)
    if (up[0] >= 0 and up[0] < region_width_global and up[1] >= 0 and up[1] < region_height_global):
        
        if up in all_nodes:
            up_node = all_nodes[up]

        else: 
            up_node = Node(up, 0)
        

        succ.append(up_node)

    down = (s[0], s[1] + 1)
    if (down[0] >= 0 and down[0] < region_width_global and down[1] >= 0 and down[1] < region_height_global):
            
            if down in all_nodes:
                down_node = all_nodes[down]

            else: 
                down_node = Node(down, 0)
            

            succ.append(down_node)


    return succ
    