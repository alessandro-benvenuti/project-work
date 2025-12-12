import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

import heapq

class Problem:
    _graph: nx.Graph
    _alpha: float
    _beta: float

    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5

        self._graph = nx.Graph()
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            self._graph.add_node(c, pos=(cities[c, 0], cities[c, 1]), gold=(1 + 999 * rng.random()))

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self._graph)

    @property
    def graph(self) -> nx.Graph:
        return nx.Graph(self._graph)

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def cost(self, path, weight):
        dist = nx.path_weight(self._graph, path, weight='dist')
        return dist + (self._alpha * dist * weight) ** self._beta

    def baseline(self):
        cost = 0
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        ).items():
            if dest == 0:
                continue
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)}) -- cost: {self.cost(path, 0):.2f}"
            )
            logging.debug(f"dummy_solution: grab {self._graph.nodes[dest]['gold']:.2f}kg of gold")
            logging.debug(
                f"dummy_solution: return to 0 ({' > '.join(str(n) for n in reversed(path))}) -- cost: {self.cost(path, self._graph.nodes[dest]['gold']):.2f}"
            )
            cost += self.cost(path, 0) + self.cost(path, self._graph.nodes[dest]['gold'])
        logging.info(f"dummy_solution: total cost: {cost:.2f}")
        return cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
    
    def solution():
        # Placeholder for user-defined solution
        pass

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
        #TODO FIXARE STA MERDA
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
    





def main():
    problem = Problem(100, density=0.2, alpha=2, beta=2)
    
    print(problem.baseline())

    cities = []
    cost = 0

    for city in range(1, 100):
        trip = Trip(cities=[(city, problem.graph.nodes[city]['gold'])], problem=problem)
        # print(trip.path)
        cost += trip.total_cost

    print(f"Total cost: {cost}")



    # print(Problem(100, density=0.2, alpha=1, beta=2).baseline())
    # print(Problem(100, density=0.2, alpha=2, beta=1).baseline())
    # print(Problem(100, density=1, alpha=1, beta=1).baseline())
    # print(Problem(100, density=1, alpha=2, beta=1).baseline())
    # print(Problem(100, density=1, alpha=1, beta=2).baseline())
    # print(Problem(1_000, density=0.2, alpha=1, beta=1).baseline())
    # print(Problem(1_000, density=0.2, alpha=2, beta=1).baseline())
    # print(Problem(1_000, density=0.2, alpha=1, beta=2).baseline())
    # print(Problem(1_000, density=1, alpha=1, beta=1).baseline())
    # print(Problem(1_000, density=1, alpha=2, beta=1).baseline())
    # print(Problem(1_000, density=1, alpha=1, beta=2).baseline())

main()