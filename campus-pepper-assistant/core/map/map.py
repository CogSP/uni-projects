# -*- coding: utf-8 -*-
from datetime import datetime
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import os
import sqlite3

class Map:
    
    """Graph-based representation of the campus layout"""
    
    def __init__(self, scale=0.3):
        # Define nodes in the campus (x, y coordinates in meters)
        self.current_position = "entrance"
        nodes = {
            'entrance':     (0,   0),
            'lobby': (-2,  0),
            # 'elevators' removed
            'bathroom':     (-3, -1.5),
            'a_corridor':   (-9,  0.8),   # A-wing corridor (north side)
            'b_corridor':   (-9, -0.8),   # B-wing corridor (south side)
            #'exit':         (-15, 0),
        }

        # A-wing rooms (classrooms + offices)
        self.A_ROOMS = [
            'A1','A2','A3','A4','A5','A6','A7',
            'A101','A102','A103','A104','A105','A106','A107','A108','A109','A110','A111','A112','A113',
            'A116','A117','A118','A119','A120','A121','A122','A125',
            'A201','A202','A203','A204','A205','A206','A207','A208','A209','A210','A-211','A212','A213','A214','A215','A216','A217','A218','A219',
            'A221','A225','A226','A227',
        ]

        # B-wing rooms (classrooms + offices)
        self.B_ROOMS = [
            'B2','B003','B007',
            'B102','B111','B113','B114','B115','B116','B117','B118','B119','B123',
            'B205','B206','B207','B208','B209','B210','B211','B212','B213','B214','B215','B216','B217','B218','B219','B220',
        ]

        # Labs / library / centers placed around the lobby
        self.OTHER_ROOMS = ['BiblioDIAG','ALCOR','BiBi','RoboticsLab','ROCOCO','B1-CIS']

       
        # A-wing: rooms along y=+2.2, corridor y=+0.8
        a_step = 0.7
        for i, code in enumerate(self.A_ROOMS):
            nodes[code] = (-6 - a_step*i, 2.2)

        # B-wing: rooms along y=-2.2, corridor y=-0.8
        b_step = 0.7
        for i, code in enumerate(self.B_ROOMS):
            nodes[code] = (-6 - b_step*i, -2.2)

        # PLACE them along the A side (same y as A rooms = 2.2), near the lobby
        other_step = 0.8
        other_start_x = -5.2           # a bit left of the lobby; adjust if you want them tighter/looser
        for i, code in enumerate(self.OTHER_ROOMS):
            nodes[code] = (other_start_x - other_step*i, 2.2)


        # Apply scale
        self.nodes = {name: (x*scale, y*scale) for name, (x, y) in nodes.items()}

        # ---- Edges ----
        self.edges = {
            'entrance': ['lobby'],
            'lobby': ['entrance', 'bathroom', 'a_corridor', 'b_corridor'],
            'bathroom': ['lobby'],
            'a_corridor': ['lobby'],
            'b_corridor': ['lobby'],
            #'exit': ['a_corridor', 'b_corridor'],
        }

        # Connect corridors directly to rooms (bidirectional)
        for code in self.A_ROOMS:
            self.edges.setdefault('a_corridor', []).append(code)
            self.edges[code] = ['a_corridor']

        for code in self.B_ROOMS:
            self.edges.setdefault('b_corridor', []).append(code)
            self.edges[code] = ['b_corridor']

        # Other rooms off the lobby
        for code in self.OTHER_ROOMS:
            self.edges.setdefault('a_corridor', []).append(code)
            self.edges[code] = ['a_corridor']

        # Location mapping
        self.location_mapping = {
            'entrance': 'entrance',
            #'exit': 'exit',
            'bathroom': 'bathroom', 'restroom': 'bathroom', 'washroom': 'bathroom',
            'reception': 'lobby', 'lobby': 'lobby',
        }
        for code in self.A_ROOMS + self.B_ROOMS + self.OTHER_ROOMS:
            self.location_mapping[code] = code


    def get_distance(self, node1, node2):
        """Calculate Euclidean distance between two nodes"""
        x1, y1 = self.nodes[node1]
        x2, y2 = self.nodes[node2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def find_shortest_path(self, end):
        """Find shortest path using Dijkstra's algorithm"""        

        start = self.current_position
        end = end

        if start not in self.nodes or end not in self.nodes:
            print("Unknown start or end node.")
            return None

        distances = {node: float('infinity') for node in self.nodes}
        distances[start] = 0
        previous = {}
        unvisited = set(self.nodes.keys())
        
        while unvisited:
            current = min(unvisited, key=lambda node: distances[node])
            if distances[current] == float('infinity'):
                break
            unvisited.remove(current)
            if current == end:
                break
            for neighbor in self.edges.get(current, []):
                if neighbor in unvisited:
                    distance = distances[current] + self.get_distance(current, neighbor)
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
        
        if end not in previous and start != end:
            return None
            
        path = []
        current = end
        while current is not None:
            path.insert(0, current)
            current = previous.get(current)
        return path if path and path[0] == start else None
    
    def draw_map_with_path(self, end, bathroom_busy, save_path='campus_map_path.png'):
        print("Drawing map with path to", end)
        path = self.find_shortest_path(end)
        print("Path found:", path)
        if not path:
            print("No path found.")
            return

        # ---------- styling knobs ----------
        PATH_LW = 3.5
        BG_EDGE_ALPHA = 0.18
        BG_NODE_ALPHA = 0.28
        BG_NODE_SIZE = 10
        PATH_NODE_SIZE = 40
        LABEL_SIZE_MAIN = 8
        LABEL_SIZE_SMALL = 7
        ZOOM_PAD = 1.2  # meters in scaled coordinates
        # -----------------------------------

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch
        try:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        except Exception:
            inset_axes = None

        # --- figure ---
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.set_facecolor('#f8f9fa')

        CORRIDORS = {'a_corridor', 'b_corridor'}
        A_SET = set(self.A_ROOMS) | set(self.OTHER_ROOMS)  # <-- include OTHER_ROOMS here
        B_SET = set(self.B_ROOMS)

        def _is_corridor_to_room(u, v):
            return ((u in CORRIDORS and (v in A_SET or v in B_SET)) or
                    (v in CORRIDORS and (u in A_SET or u in B_SET)))

        def _elbow_points(u, v):
            (x1, y1), (x2, y2) = self.nodes[u], self.nodes[v]
            if u in CORRIDORS and (v in A_SET or v in B_SET):
                return [(x1, y1), (x2, y1), (x2, y2)]
            if v in CORRIDORS and (u in A_SET or u in B_SET):
                return [(x1, y1), (x1, y2), (x2, y2)]
            return [(x1, y1), (x2, y2)]

        
        # background edges
        for u, nbrs in self.edges.items():
            for v in nbrs:
                if _is_corridor_to_room(u, v):
                    p1, p2, p3 = _elbow_points(u, v)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', alpha=BG_EDGE_ALPHA, lw=1, zorder=1)
                    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color='black', alpha=BG_EDGE_ALPHA, lw=1, zorder=1)
                else:
                    (x1, y1), (x2, y2) = self.nodes[u], self.nodes[v]
                    ax.plot([x1, x2], [y1, y2], color='black', alpha=BG_EDGE_ALPHA, lw=1, zorder=1)

        # path arrows
        def _arrow(p, q, color='red'):
            if _is_corridor_to_room(p, q):
                a, b, c = _elbow_points(p, q)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=PATH_LW, zorder=5)
                arr = FancyArrowPatch(b, c, arrowstyle='-|>', mutation_scale=12, lw=PATH_LW, color=color, zorder=6)
                ax.add_patch(arr)
            else:
                (x1, y1), (x2, y2) = self.nodes[p], self.nodes[q]
                arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12, lw=PATH_LW, color=color, zorder=6)
                ax.add_patch(arr)


        for i in range(len(path) - 1):
            _arrow(path[i], path[i+1])

        # --- start & end markers ---
        start_node = self.current_position
        end_node = end
        sx, sy = self.nodes[start_node]
        ex, ey = self.nodes[end_node]

        ax.scatter(sx, sy, s=120, color='#2ca02c', zorder=6, label='Start')  # green
        end_color = '#d62728' if (end_node == "bathroom" and bathroom_busy) else '#1f77b4'
        ax.scatter(ex, ey, s=120, color=end_color, zorder=6, label='End')

        # --- minimal labeling: backbone + path + end ---
        special_labels = {'entrance', 'lobby', 'a_corridor', 'b_corridor', 'bathroom'}
        to_label = set(path) | {end_node} | special_labels

        def _pretty(lbl):
            return lbl.replace('_entrance', '').replace('_', ' ').title()

        # Per-label nudges (in axis units after scaling)
        label_offsets = {
            'a_corridor': (0.25, 0.55),   # move right and clearly above the corridor
            # Optional tweaks if you like:
            # 'b_corridor': (0.25, -0.55),
            # 'lobby': (0.20, 0.25),
            # 'bathroom': (0.20, -0.25),
        }
        default_offset = (0.12, 0.12)

        for node in to_label:
            if node not in self.nodes:
                continue
            x, y = self.nodes[node]
            dx, dy = label_offsets.get(node, default_offset)
            ax.text(
                x + dx, y + dy, _pretty(node),
                fontsize=(LABEL_SIZE_MAIN if node in special_labels or node in {start_node, end_node}
                        else LABEL_SIZE_SMALL),
                color='black', zorder=7, ha='left', va='bottom'
            )

        # --- zoom to the path bbox with padding ---
        xs, ys = zip(*[self.nodes[n] for n in path])
        xmin, xmax = min(xs) - ZOOM_PAD, max(xs) + ZOOM_PAD
        ymin, ymax = min(ys) - ZOOM_PAD, max(ys) + ZOOM_PAD
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.axis('equal')
        ax.axis('off')
        ax.legend(loc='lower left', fontsize=8, frameon=False)

        # # --- tiny overview inset (optional) ---
        # if inset_axes is not None:
        #     iax = inset_axes(ax, width="20%", height="35%", loc='upper right', borderpad=1.2)
        #     iax.set_facecolor('#ffffff')
        #     # background edges
        #     for u, nbrs in self.edges.items():
        #         for v in nbrs:
        #             (x1, y1), (x2, y2) = self.nodes[u], self.nodes[v]
        #             iax.plot([x1, x2], [y1, y2], color='black', alpha=0.12, lw=0.8, zorder=1)
        #     # path
        #     px, py = zip(*[self.nodes[n] for n in path])
        #     iax.plot(px, py, color='red', lw=2, zorder=3)
        #     iax.scatter([sx, ex], [sy, ey], s=30, color=['#2ca02c', end_color], zorder=4)
        #     iax.axis('equal'); iax.axis('off')

        plt.tight_layout()

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_path = os.path.join(base_dir, "tablet/img", save_path)
        print("Saving map to", save_path)
        plt.savefig(save_path, dpi=300, transparent=True)
        plt.close()
        print("Map saved to ", save_path)
        return path
