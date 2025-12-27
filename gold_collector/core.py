import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import heapq

class Trip:
    def __init__(self, cities: list, problem: Problem):
        self.cities = cities        # tuples of (city_index, gold_amount)
        self.path = self.compute_optimal_path(problem)
        self.total_cost = self.compute_cost(problem)
        self.total_gold = sum([gold for _, gold in cities])

    def dijkstra_with_gold(self, graph: nx.Graph, source: int, target: int, gold:float, problem: Problem):
        """
        computes the optimal path from source to target in graph considering gold pickup and cost function.:
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

    def compute_optimal_path(self, problem: Problem):
        path = []
        for i, x in enumerate(self.cities):
            city = x[0]

            assert city in problem.graph.nodes, f"City {city} not in graph"
            if i == 0:
                source = 0
                previous_gold = 0
            else:
                source = self.cities[i-1][0]
                previous_gold += self.cities[i-1][1]
            
            # p = nx.shortest_path(problem.graph, source=source, target=city, weight='dist')
            p = self.dijkstra_with_gold(problem.graph, source=source, target=city, gold=previous_gold, problem=problem)
            
            for node in p[0:-1]:
                path.append((node, previous_gold))

        previous_gold += self.cities[-1][1]
        p = self.dijkstra_with_gold(problem.graph, source=self.cities[-1][0], target=0, gold=previous_gold, problem=problem)
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
