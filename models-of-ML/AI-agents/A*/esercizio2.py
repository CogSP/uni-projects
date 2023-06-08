
def successors(s, is_solid, region_width, region_height):
    
    right = (s[0]+1, s[1])
    left = (s[0]-1, s[1])
    down = (s[0], s[1]+1)
    up = (s[0], s[1]-1)

    rosa = [right, left, down, up] 
    succs = []
    for dir in rosa:
        # se siamo fuori dai bordi la cella non va ovviamente considerata
        if dir[0] < 0 or dir[0] >= region_width or dir[1] < 0 or dir[1] >= region_height:
          continue
    
        # se la cella non è solida classico costo di 1
        if is_solid(dir) == False:
            succs.append((dir, 1)) 
        # se la cella è solida gli mettiamo un costo altissimo
        elif is_solid(dir) == True:
          succs.append((dir, 100)) 
            
    return succs


# Commento per motivare la scelta dell'euristica:
# Come ho già detto per l'esercizio 1, ho utilizzato la distanza di Manhattan poiché
# la possibilità di muoversi solo in orizzontale e verticale lo consigliava

# Un ulteriore aggiunta che si potrebbe fare, poiché in questo esercizio alle celle
# solide è associato un peso alto, mentre alle celle non solide un peso basso
# è quello di pesare la distanza sulla base della cella in input all'euristica.

def heuristic(s, goal, is_solid, region_width, region_height):

   
    h_s = abs(s[0] - goal[0]) + abs(s[1] - goal[1]) 
    return h_s

