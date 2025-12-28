from icecream import ic
from copy import copy
import networkx as nx
import logging
from itertools import combinations

from gold_collector.core import Solution, Trip
from gold_collector.utils import generate_baseline, generate_smart_baseline, generate_topology_savings


def solve(problem: Problem) -> Solution:
    # Pre-compute Distance Matrix ONCE (O(N*E log V))
    dist_matrix = dict(nx.all_pairs_dijkstra_path_length(problem._graph, weight='dist'))

    base = generate_smart_baseline(problem)
    print(f"Baseline total cost: {base.total_cost}")
    sol = generate_topology_savings(problem)
    print(f"Topology Savings total cost: {sol.total_cost}")
