import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

import os
import csv
import time
import itertools

from src.core import Trip, Solution
from src.solve import solve

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
    return solve(problem)[0]
    



def main():
    num_cities_list = [50, 100, 200, 1000]
    num_cities_list = [1000]
    densities = [0.1, 0.2, 0.5, 1.0]
    alphas = [0.1, 1, 2, 5]
    betas = [0.1, 1, 2, 5]

    # Output file setup
    os.makedirs("logs", exist_ok=True)
    csv_file = "logs/results.csv"
    
    # Define columns
    columns = [
        "num_cities", "density", "alpha", "beta", 
        "baseline_cost", "solution_cost", "improvement_percent", "time_taken_sec",
        "errors"
    ]

    # Create file and write header if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    print(f"--- Starting Experiments ---")
    print(f"Results will be appended to: {csv_file}")
    
    # Generate all combinations
    combinations = list(itertools.product(num_cities_list, densities, alphas, betas))
    total_runs = len(combinations)
    
    for i, (n, d, a, b) in enumerate(combinations):
        print(f"\n[{i+1}/{total_runs}] Running: Cities={n}, Dens={d}, Alpha={a}, Beta={b}")
        
        try:
            # Problem  Setup
            problem = Problem(num_cities=n, density=d, alpha=a, beta=b)
            
            # Teacher Baseline
            baseline_cost = problem.baseline()
            
            # My Solution
            start_time = time.time()
            solution_path, solution_cost, is_valid, message = solve(problem)
            elapsed = time.time() - start_time

            # Handle invalid solutions
            if not is_valid:
                print(f"   -> SOLUTION INVALID: {message}")
                # Log invalid solutions with a special marker
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([n, d, a, b, f"{baseline_cost:.2f}", "INVALID", "0", f"{elapsed:.2f}", f"{message}"])
                continue
            
            # 4. Calculate Statistics
            if baseline_cost > 0:
                improvement = ((baseline_cost - solution_cost) / baseline_cost) * 100
            else:
                improvement = 0.0

            # 5. Log to CSV
            row = [n, d, a, b, f"{baseline_cost:.2f}", f"{solution_cost:.2f}", f"{improvement:.2f}", f"{elapsed:.2f}", "-"]
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            print(f"   -> Result: Base={baseline_cost:.0f} | Sol={solution_cost:.0f} | Imp={improvement:.2f}% | Time={elapsed:.1f}s")

        except Exception as e:
            print(f"   -> ERROR: {e}")
            #  Log errors to the CSV or a separate file
            with open("logs/errors.csv", mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n, d, a, b, "ERROR", "ERROR", "0", "0", f"{str(e)}"])

    # problem = Problem(100, density=0.2, alpha=1, beta=1)

    # baseline = problem.baseline()
    # print(f"Teacher baseline: {baseline}")
    

    # sol, cost = solution(problem)
    # print(f"Solution total cost: {cost}")

    # improvement = (baseline - cost) / baseline * 100
    # print(f"Improvement over baseline: {improvement:.2f}%")

    # times_better = baseline / cost
    # print(f"Times better than baseline: {times_better:.2e}")

    # with open("logs/solution.txt", "w") as f:
    #     f.write(str(sol))


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