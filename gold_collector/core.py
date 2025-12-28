import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import heapq

class Trip:
    def __init__(self, cities: list, problem, precomputed_path=None, static_paths=None):
        """
        :param cities: List of (city_index, gold_amount)
        :param precomputed_path: Full path list (for baseline generation)
        :param static_paths: Dict {'outbound': [nodes], 'inbound': [nodes]} 
                             containing precomputed shortest paths for legs.
        """
        self.cities = cities

        if precomputed_path:
            # FASTEST: Use the full provided path (e.g. for Initial Baseline)
            self.path = precomputed_path
        else:
            # HYBRID: Compute path, potentially reusing static segments
            self.path = self.compute_optimal_path(problem, static_paths)

        self.total_cost = self.compute_cost(problem)
        self.total_gold = sum([gold for _, gold in cities])

    

    def dijkstra_with_gold(self, graph: nx.Graph, source: int, target: int, gold:float, problem: Problem):
        """
        Computes the optimal path from source to target in graph considering gold pickup and cost function:
        dist + (alpha * dist * total_gold) ** beta
        """
        pq = [(0.0, source)]
        dist = {source: 0.0}
        prev = {}
        visited = set()

        while pq:
            current_dist, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == target:
                break

            for v in graph.neighbors(u):
                edge_weight = graph[u][v]['dist']
                step = edge_weight + (problem.alpha * edge_weight * gold) ** problem.beta
                new_dist = current_dist + step

                if new_dist < dist.get(v, float('inf')):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        if target not in dist:
            raise nx.NetworkXNoPath(f"No path from {source} to {target}")

        # reconstruct path
        path = [target]
        u = target
        while u != source:
            u = prev[u]
            path.append(u)
        path.reverse()
        return path
    
    def astar_with_gold(self, graph: nx.Graph, source: int, target: int, gold: float, problem):
        """
        Computes the optimal path using A* search.
        Heuristic: The cost function applied to the Euclidean distance.
        """
        
        # 1. Define the Heuristic Function (Local helper or separate method)
        def heuristic(u, v):
            # Get positions
            pos_u = np.array(graph.nodes[u]['pos'])
            pos_v = np.array(graph.nodes[v]['pos'])
            
            # Euclidean distance (Straight line)
            dist = np.linalg.norm(pos_u - pos_v)
            
            # The minimum possible cost to travel this distance with current gold
            return dist + (problem.alpha * dist * gold) ** problem.beta

        # 2. Initialize A* structures
        # Priority Queue stores: (f_score, current_node)
        # f_score = g_score (actual cost so far) + h_score (estimated cost to go)
        start_h = heuristic(source, target)
        pq = [(start_h, source)]
        
        # g_score: The cheapest cost found so far to reach a node
        g_score = {source: 0.0}
        
        prev = {}
        visited = set()

        while pq:
            # Pop node with lowest ESTIMATED total cost
            current_f, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)

            if u == target:
                break

            # If the popped f_score is worse than what we already found, skip (lazy deletion)
            # Note: We can't strictly rely on this optimization with non-monotonic heuristics,
            # but ours is monotonic, so it's safe.
            if current_f > g_score.get(u, float('inf')) + heuristic(u, target):
                continue

            for v in graph.neighbors(u):
                # Calculate actual cost of this step
                edge_weight = graph[u][v]['dist']
                step_cost = edge_weight + (problem.alpha * edge_weight * gold) ** problem.beta
                
                tentative_g = g_score[u] + step_cost

                if tentative_g < g_score.get(v, float('inf')):
                    # We found a better path to v!
                    prev[v] = u
                    g_score[v] = tentative_g
                    
                    # f = g + h
                    f_score = tentative_g + heuristic(v, target)
                    heapq.heappush(pq, (f_score, v))

        if target not in g_score:
             # Fallback or error if graph is disconnected
            raise nx.NetworkXNoPath(f"No path from {source} to {target}")

        # Reconstruct path
        path = [target]
        curr = target
        while curr != source:
            curr = prev[curr]
            path.append(curr)
        path.reverse()
        
        return path

    def compute_optimal_path(self, problem, static_paths=None):
        path = []
        
        # 1. OUTBOUND LEG (Base -> First City)
        # The truck is empty (gold=0), so Shortest Distance is ALWAYS optimal.
        # We use the precomputed path regardless of Beta.
        if static_paths and 'outbound' in static_paths:
            # Append everything except the destination (handled in loop)
            for node in static_paths['outbound'][:-1]:
                path.append((node, 0.0))
        else:
            # Fallback if no static path provided
            first_city = self.cities[0][0]
            p = self.astar_with_gold(problem.graph, 0, first_city, 0.0, problem)
            for node in p[0:-1]:
                path.append((node, 0.0))

        # 2. INTERMEDIATE LEGS (City -> City)
        # Must always be computed dynamically because weight varies.
        previous_gold = 0
        for i, x in enumerate(self.cities):
            city = x[0]
            current_gold = x[1]
            
            if i > 0:
                source = self.cities[i-1][0]
                p = self.astar_with_gold(problem.graph, source, city, previous_gold, problem)
                for node in p[0:-1]:
                    path.append((node, previous_gold))
            
            previous_gold += current_gold

        # 3. INBOUND LEG (Last City -> Base)
        # Here we apply your logic:
        # If Beta <= 1: Precomputed (Distance) is optimal. Use it.
        # If Beta > 1: Precomputed is risky. Use A* to optimize cost.
        if static_paths and 'inbound' in static_paths and problem.beta <= 1:
            for node in static_paths['inbound']:
                path.append((node, previous_gold))
        else:
            # Run A* / Dijkstra because Beta is high (or no static path available)
            source = self.cities[-1][0]
            p = self.dijkstra_with_gold(problem.graph, source, 0, previous_gold, problem)
            for node in p:
                path.append((node, previous_gold))

        return path


    def compute_cost(self, problem: Problem):
        cost = 0
        total_gold = 0
        for i, x in enumerate(self.path):
            dist = problem.graph[self.path[i - 1][0]][x[0]]['dist'] if i > 0 else 0
            cost += dist + (problem.alpha * dist * total_gold) ** problem.beta
            total_gold = x[1]
        return cost
    
class Solution:
    def __init__(self, trips: list[Trip]):
        self.trips = trips
        self.total_cost = sum([trip.total_cost for trip in trips])
        self.total_gold = sum([trip.total_gold for trip in trips])

    

    def combine_trips(trip1: Trip, trip2: Trip, problem: Problem) -> Trip:
        combined_cities = trip1.cities + trip2.cities
        combined_cities.sort(key=lambda x: x[1])
        combined_trip = Trip(cities=combined_cities, problem=problem)
        return combined_trip
