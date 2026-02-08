import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

import heapq

from gold_collector.core import Trip, Solution
from gold_collector.solve import solve

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

        self.num_cities = num_cities

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
        total_cost = 0
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        ).items():
            cost = 0
            for c1, c2 in zip(path, path[1:]):
                cost += self.cost([c1, c2], 0)
                cost += self.cost([c1, c2], self._graph.nodes[dest]['gold'])
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)} ({cost})"
            )
            total_cost += cost
        return total_cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
    

# --- MONKEY PATCH START ---
# We overwrite the 'graph' property of the Problem class.
# Instead of creating a copy (nx.Graph(self._graph)), we return the private reference directly.
def fast_graph_accessor(self):
    return self._graph          # This way we do not create a copy each time, making it extremely faster.

Problem.graph = property(fast_graph_accessor)
# --- MONKEY PATCH END ---
    
def solution(problem: Problem) -> Solution:
    return solve(problem)
    



def main():
    problem = Problem(1000, density=0.2, alpha=1, beta=2)

    baseline = problem.baseline()
    print(f"Teacher baseline: {baseline}")
    

    sol, cost = solution(problem)
    print(f"Solution total cost: {cost}")

    improvement = (baseline - cost) / baseline * 100
    print(f"Improvement over baseline: {improvement:.2f}%")

    times_better = baseline / cost
    print(f"Times better than baseline: {times_better:.2e}")

    # print(f"--- SOLUTION PATH ---\n{sol}")

    with open("logs/result.txt", "w") as f:
        f.write(str(sol))


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

if __name__ == '__main__':
    main()