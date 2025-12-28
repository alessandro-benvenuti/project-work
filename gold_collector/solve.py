from icecream import ic
from copy import copy
import networkx as nx
import logging
from itertools import combinations

from gold_collector.core import Solution, Trip
from gold_collector.utils import generate_baseline, generate_topology_savings, generate_split_visits, generate_random_chunk_visits, generate_adaptive_split


def solve(problem: Problem) -> Solution:
    # The problem will be solved by using simulated annealing, but the solution will be different based on the value of Alpha and Beta.

    # At first check Alpha: if its value is very low (<0.002), then the problem is almost linear.
    # In this case it is better to merge paths rather than splitting them, even if Beta is high, 
    # since the power of a number below 1 is going to decrease its value.
    # But why 0.002? Because with Alpha=0.002 and max distance ~1 and max gold ~1000, (mean distance ~0.5, mean gold ~500) the cost term becomes:
    # cost = dist + (0.002 * dist * 500) ** Beta
    # which is approximately dist + (1 * dist) ** Beta
    # So for Beta=1, cost = dist + dist = dist (dominated by distance)
    # For Beta=2, cost = dist + (dist)^2 = dist + dist^2 (still dominated by distance)
    # For Beta=3, cost = dist + (dist)^3 = dist + dist^3 (still dominated by distance)
    # And remeber that distance < 1, so dist^2 and dist^3 are even smaller.
    # Therefore, Alpha=0.002 is a reasonable threshold to consider the problem as distance-dominated.
    
    # If Beta <= 1: We can rely on precomputed shortest paths (geometric distance) since cost is dominated by distance.
    # Also the goal of the simulated annealing will be to try merging and splitting trips (or mixing them).

    # If Beta > 1: The shortest path is no longer necessarly optimal due to the exponential cost of carrying gold.
    # The goal is shifted to finding the best way to split the gold collection into multiple trips to minimize the cost.
    # Each trip will be optimized individually using A* with a cost function that considers both distance and gold carried.

    if problem.alpha <= 0.002:
        # Use precomputed shortest paths (geometric distance)
        sol = generate_topology_savings(problem)
    
    else:
        if problem.beta <= 1:
            # Use precomputed shortest paths (geometric distance)
            sol = generate_topology_savings(problem)

        else:
            # Use adaptive splitting strategy to minimize cost with high Beta
            # This involves dynamically computing paths with A* for each trip
            sol = generate_adaptive_split(problem)
    
    
    return sol
