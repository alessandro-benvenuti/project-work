from icecream import ic
from copy import copy, deepcopy
import networkx as nx
import logging
from itertools import combinations
import math
import random

from gold_collector.core import Solution, Trip
from gold_collector.utils import generate_baseline, generate_topology_savings, generate_split_visits, generate_random_chunk_visits, generate_adaptive_split

class AdaptiveSelector:
    def __init__(self, alpha):
        # 1. Biased Initialization based on your Alpha discovery
        if alpha <= 0.001:
            self.weights = {"split": 10.0, "merge": 50.0, "swap": 20.0}
        else:
            self.weights = {"split": 10.0, "merge": 10.0, "switch": 10.0}
            
        self.scores = {k: 0 for k in self.weights}
        self.usage_count = {k: 0 for k in self.weights}
        self.decay = 0.8  # How fast to forget past success

    def select_operator(self, valid_ops):
        # Filter weights to only valid operators
        relevant_weights = {k: self.weights[k] for k in valid_ops if k in self.weights}
        
        total = sum(relevant_weights.values())
        
        # Safety fallback if something goes wrong (e.g. total is 0)
        if total == 0:
            return random.choice(valid_ops)
            
        probs = [w / total for w in relevant_weights.values()]
        choice = random.choices(list(relevant_weights.keys()), weights=probs, k=1)[0]
        
        self.usage_count[choice] += 1
        return choice

    def update_score(self, operator, outcome_type):
        """
        outcome_type:
        3 = Global Best (New Record)
        2 = Local Improvement
        1 = Accepted (Worse but accepted by SA temp)
        0 = Rejected
        """
        rewards = {3: 5.0, 2: 2.0, 1: 0.5, 0: 0}
        self.scores[operator] += rewards[outcome_type]

    def update_weights(self):
        """Call this every N iterations"""
        for op in self.weights:
            if self.usage_count[op] > 0:
                # Average score per usage
                avg_score = self.scores[op] / self.usage_count[op]
                
                # Apply decay: (History * 0.8) + (Recent Performance * 0.2)
                self.weights[op] = (self.weights[op] * self.decay) + \
                                   (avg_score * (1 - self.decay))
                
                # Reset tracking for next batch
                self.scores[op] = 0
                self.usage_count[op] = 0
                
            # Ensure weight never drops to absolute zero (keep tiny probability)
            self.weights[op] = max(self.weights[op], 1.0)

def simulated_annealing(initial_solution: Solution, problem, initial_temp: float = 1000.0, cooling_rate: float = 0.995, max_iterations: int = 10000) -> Solution:

    adaptive_selector = AdaptiveSelector(alpha=problem.alpha)

    current_solution = initial_solution
    current_cost = current_solution.total_cost
    best_solution = current_solution
    best_cost = current_cost
    temperature = initial_temp

    for iteration in range(max_iterations):
        # Create a neighbor solution by modifying the current solution
        neighbor_solution = deepcopy(current_solution)

        # Determine Valid Operators for this specific state
        valid_operators = []
        # Check basic constraints
        has_multiple_trips = len(current_solution.trips) > 1
        has_complex_trip = any(len(t.cities) > 1 for t in current_solution.trips)
        
        if has_multiple_trips:
            valid_operators.append('merge')
            
        if has_complex_trip:
            valid_operators.append('split')
            
        if has_multiple_trips and has_complex_trip:     # Only allow swap if it's meaningful (avoiding 1-city vs 1-city null swaps)
            valid_operators.append('swap')

        if not valid_operators:
             # Stop early or break, as no moves are possible
            break

        # Choose a modification: merge trips, split trips, or swap cities between trips
        modification_type = adaptive_selector.select_operator(valid_ops=valid_operators)

        did_modify = False

        if modification_type == 'merge' and len(neighbor_solution.trips) > 1:
            trip1, trip2 = random.sample(neighbor_solution.trips, 2)
            merged_cities = trip1.cities + trip2.cities
            new_trip = Trip(merged_cities, problem)
            neighbor_solution.trips.remove(trip1)
            neighbor_solution.trips.remove(trip2)
            neighbor_solution.trips.append(new_trip)
            did_modify = True

        elif modification_type == 'split' and len(neighbor_solution.trips) > 0:
            trip = random.choice(neighbor_solution.trips)
            if len(trip.cities) > 1:
                split_index = random.randint(1, len(trip.cities) - 1)
                trip1_cities = trip.cities[:split_index]
                trip2_cities = trip.cities[split_index:]
                trip1 = Trip(trip1_cities, problem)
                trip2 = Trip(trip2_cities, problem)
                neighbor_solution.trips.remove(trip)
                neighbor_solution.trips.append(trip1)
                neighbor_solution.trips.append(trip2)
            did_modify = True

        elif modification_type == 'swap' and len(neighbor_solution.trips) > 1:
            trip1, trip2 = random.sample(neighbor_solution.trips, 2)
            if len(trip1.cities) > 0 and len(trip2.cities) > 0:
                city1_index = random.randint(0, len(trip1.cities) - 1)
                city2_index = random.randint(0, len(trip2.cities) - 1)
                trip1.cities[city1_index], trip2.cities[city2_index] = trip2.cities[city2_index], trip1.cities[city1_index]
                trip1.path = trip1.compute_optimal_path(problem)
                trip1.total_cost = trip1.compute_cost(problem)
                trip2.path = trip2.compute_optimal_path(problem)
                trip2.total_cost = trip2.compute_cost(problem)
            did_modify = True

        # Evaluate the neighbor solution
        neighbor_cost = sum(trip.total_cost for trip in neighbor_solution.trips)
        cost_diff = neighbor_cost - current_cost


        outcome = 0 # Default: Rejected
        
        # Accept if better OR with probability exp(-delta/T)
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            
            if cost_diff < 0:
                outcome = 2 # Local Improvement
            else:
                outcome = 1 # Accepted worse move (Exploration)

            # Update Global Best
            if current_cost < best_cost:
                best_solution = deepcopy(current_solution) # Save a snapshot!
                best_cost = current_cost
                outcome = 3 # Global Record!
        
        # Tell the selector how well the operator performed
        adaptive_selector.update_score(modification_type, outcome)
        
        # Update weights periodically (e.g., every 100 steps)
        if iteration % 100 == 0:
            adaptive_selector.update_weights()

        
        # Cool down the temperature
        temperature *= cooling_rate

    return best_solution


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

    sol = simulated_annealing(sol, problem, initial_temp=5000.0, cooling_rate=0.995, max_iterations=5000)
    
    
    return sol
