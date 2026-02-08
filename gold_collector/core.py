import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import heapq

class Trip:
    def __init__(self, cities: list, problem, times_taken=1, path_matrix=None, use_precompute=False, use_dijkstra=False):
        """
        :param cities: List of (city_index, gold_amount)
        :param path_matrix: Dict or Matrix [source][target] -> List of nodes (path)
        :param use_precompute: Boolean flag. If True, use path_matrix for routing.
        :param use_dijkstra: Boolean flag. If True, force Dijkstra algorithm for exact path calculation.
        """
        self.cities = cities
        self.total_gold = sum([gold for _, gold in cities])
        self.times_taken = times_taken
        
        # Determine strategy
        if use_precompute and path_matrix is not None:
            # FAST MODE: Use precomputed geometric paths
            self.path = self._build_path_from_matrix(path_matrix)
        else:
            # ACCURATE MODE: Run A* dynamically (slow but exact for final verification)
            self.path = self.compute_optimal_path(problem, use_dijkstra=use_dijkstra)
            
        self.total_cost = self.compute_cost(problem) * self.times_taken

    def change_times_taken(self, new_times):
        if new_times <= 0:
            # Handle the case where we reduce it to 0 (remove trip)
            self.times_taken = 0
            self.total_cost = 0
            return self
            
        # Update cost proportionally
        # (Assuming the path doesn't change, just the number of repetitions)
        unit_cost = self.total_cost / self.times_taken
        self.total_cost = unit_cost * new_times
        self.times_taken = new_times
        return self

    
    def _build_path_from_matrix(self, path_matrix):
        """
        Stitches together the full route using the precomputed segments.
        """
        full_path = []
        current_gold = 0.0
        
        # 1. Base -> First City (Empty)
        start_node = 0
        first_city = self.cities[0][0]
        
        # Get path from 0 to first_city
        segment = path_matrix[start_node][first_city]
        # Append all except last (to avoid duplicate nodes)
        for node in segment[:-1]:
            full_path.append((node, 0.0))
            
        # 2. City -> City
        for i, (city, gold) in enumerate(self.cities):
            current_gold += gold
            
            # Determine destination (Next city or Base)
            if i < len(self.cities) - 1:
                next_city = self.cities[i+1][0]
            else:
                next_city = 0 # Return to base
            
            # Get precomputed segment
            try:
                segment = path_matrix[city][next_city]
            except KeyError:
                # Fallback if matrix is incomplete (should not happen in connected graph)
                segment = [city, next_city]

            # Append segment
            # If it's the final return to base, include the last node (0)
            # Otherwise, skip the last node because the next iteration starts with it
            slice_end = None if next_city == 0 else -1
            
            for node in segment[:slice_end]:
                full_path.append((node, current_gold))

        return full_path

    

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

    def compute_optimal_path(self, problem, use_dijkstra=False):
        path = []
        
        # 1. Base -> First City
        # Empty truck: Shortest path is always optimal
        first_city = self.cities[0][0]
        
        # Use Dijkstra if requested, otherwise A*
        if use_dijkstra:
            # You already have dijkstra_with_gold in your class
            p = self.dijkstra_with_gold(problem.graph, 0, first_city, 0.0, problem)
        else:
            p = self.astar_with_gold(problem.graph, 0, first_city, 0.0, problem)
            
        for node in p[:-1]:
            path.append((node, 0.0))

        # 2. City -> City
        current_gold = 0.0
        for i, (city, gold) in enumerate(self.cities):
            current_gold += gold
            
            source = city
            target = self.cities[i+1][0] if i < len(self.cities)-1 else 0
            
            if use_dijkstra:
                p = self.dijkstra_with_gold(problem.graph, source, target, current_gold, problem)
            else:
                p = self.astar_with_gold(problem.graph, source, target, current_gold, problem)
            
            # Slice and append
            slice_p = p if target == 0 else p[:-1]
            for node in slice_p:
                path.append((node, current_gold))
                
        return path


    def compute_cost(self, problem: Problem):
        """Calculates exact cost by traversing the path edges."""
        cost = 0.0
        # self.path is list of (node_id, gold_on_truck)
        for i in range(1, len(self.path)):
            u = self.path[i-1][0]
            v = self.path[i][0]
            w = self.path[i-1][1] # Gold carried on this edge
            
            # Fast lookup
            dist = problem.graph[u][v]['dist']
            
            # The Formula
            cost += dist + (problem.alpha * dist * w) ** problem.beta
            
        return cost

        # cost = 0
        # total_gold = 0
        # for i, x in enumerate(self.path):
        #     dist = problem.graph[self.path[i - 1][0]][x[0]]['dist'] if i > 0 else 0
        #     cost += dist + (problem.alpha * dist * total_gold) ** problem.beta
        #     total_gold = x[1]
        # return cost
    
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
