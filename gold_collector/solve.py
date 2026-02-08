from concurrent.futures import ProcessPoolExecutor
from icecream import ic
import numpy as np
from copy import copy, deepcopy
import networkx as nx
import logging
from itertools import combinations
import math
import random
import os

from gold_collector.core import Solution, Trip
from gold_collector.utils import generate_baseline, generate_topology_savings, generate_split_visits, generate_random_chunk_visits, generate_adaptive_split, compute_distance_matrix, precompute_weighted_paths
from gold_collector.genetic import Island, run_island_evolution

class Archipelago:
    def __init__(self, problem, num_islands=4, population_size=50, offspring_size=20):
        self.problem = problem
        self.islands = []
        self.distance_matrix = compute_distance_matrix(problem)

        # Determine if we should use the heuristic precomputation based on problem size and beta:
        # practical threshold can be tuned based on experiments. Here we choose num_cities > 200 as a heuristic cutoff for when to use the approximation.
        # Also if beta <= 1 then the cost is more distance-sensitive, so we can afford to precompute paths in any case to speeed up the search.
        self.use_precompute = (problem.beta <= 1.0) or (problem.num_cities > 200)
        
        if self.use_precompute:
            print("--- Precomputing Paths (Heuristic Mode) ---")
            self.path_matrix = precompute_weighted_paths(problem, ref_gold_ratio=0.5)
        else:
            self.path_matrix = None

        strategies = ["random", "merge", "adjust_repeats", "merge"]

        # Initialize islands with different strategies or seed solutions
        for i in range(num_islands):
            # Each island gets a different seed solution
            if i==0:
                seed_solution = generate_baseline(problem, path_matrix=self.path_matrix, use_precompute=self.use_precompute)
            elif i==1:
                seed_solution = generate_topology_savings(problem, path_matrix=self.path_matrix, use_precompute=self.use_precompute)
            else:
                seed_solution = generate_adaptive_split(problem, path_matrix=self.path_matrix, use_precompute=self.use_precompute, max_search=1000)

            island = Island(
                island_id=i,
                strategy=strategies[i % len(strategies)],
                seed_solution=seed_solution,
                problem=problem,
                path_matrix=self.path_matrix,
                use_precompute=self.use_precompute,
                population_size=population_size,
                offspring_size=offspring_size
            )
            self.islands.append(island)

    def perform_migration(self):
        """
        Executes Ring Migration: 
        Island 0 -> Island 1 -> Island 2 -> ... -> Island 0
        The Best individual of Island i replaces the Worst of Island i+1.
        """
        # 1. Harvest the Elites
        # We must Deep Copy so mutations in the new island don't affect the old one
        # migrants[i] is the best solution from island i
        migrants = [deepcopy(island.population[0]) for island in self.islands]

        # 2. Inject into Neighbors
        for i in range(len(self.islands)):
            # Determine neighbor index (Ring Topology)
            target_idx = (i + 1) % len(self.islands)
            target_island = self.islands[target_idx]

            # Replace the worst individual (last in sorted list) with the incoming elite
            # Note: We assume population is sorted by cost (Ascending: Best -> Worst)
            target_island.population[-1] = migrants[i]

            # Re-sort the target island immediately to maintain invariant
            target_island.population.sort(key=lambda x: x.cost)

        # Optional: Log the migration event
        best_costs = [isl.population[0].cost for isl in self.islands]
        print(f"--- Migration Complete. Island Bests: {['{:.2f}'.format(c) for c in best_costs]} ---")



    def run_parallel(self, total_generations=200, migration_interval=20):
        # --- LOGGING SETUP ---
        os.makedirs("logs", exist_ok=True)
        # Initialize/Clear the checkpoint file
        with open("logs/checkpoint.txt", "w") as f:
            f.write(f"STARTING EVOLUTION: {len(self.islands)} Islands, {self.islands[0].pop_size} Pop/Island\n")

        # We run in blocks (epochs)
        num_epochs = total_generations // migration_interval
        
        with ProcessPoolExecutor(max_workers=len(self.islands)) as executor:
            for epoch in range(num_epochs):
                print(f"--- Epoch {epoch+1}/{num_epochs} (Running parallel) ---")
                
                # 1. SCATTER: Send islands to workers
                # executor.map runs 'run_island_evolution' on each island in parallel
                futures = [executor.submit(run_island_evolution, island, migration_interval) 
                           for island in self.islands]
                
                # 2. GATHER: Collect results back to main process
                self.islands = [f.result() for f in futures]
                
                # 3. MIGRATE: Main process shuffles genes
                self.perform_migration()

                # 4. REPORT & LOG
                best_ind = min((isl.population[0] for isl in self.islands), key=lambda x: x.cost)
                print(f"Epoch {epoch+1}/{num_epochs} - Global Best: {best_ind.cost:.2f}")

                # 4.1. Append to Checkpoint
                with open("logs/checkpoint.txt", "a") as f:
                    f.write(f"Epoch {epoch+1}: Cost {best_ind.cost}\n")
                    for trip in best_ind.trips:
                        f.write(f"  Trip: {[c[0] for c in trip.cities]} | Cost: {trip.total_cost}\n")
                
                

        best_ones = [isl.population[0] for isl in self.islands]
        best_solution = min(best_ones, key=lambda ind: ind.cost)
        final_solution = best_solution.to_solution(self.problem)

        # Extract the full path from the solution
        cities = solution_to_cities(final_solution)

        # Now we check if this final list is actually valid
        is_valid, message = verify_delta_solution(cities, self.problem, tolerance=1e-3)

        if not is_valid:
            print(f"FINAL CHECK FAILED: {message}")
            # Option: Return empty or raise error
        else:
            print(f"FINAL CHECK PASSED")

        return cities, final_solution.total_cost, is_valid, message
        

def solution_to_cities(solution: Solution):
        """
        Converts a Solution object (with Trips and Paths) into the required output format:
        List of (node_id, gold_picked_up_here) in the order they are visited.
        """
        # Extract the full path from the solution
        cities = []
        for trip in solution.trips:
            # Convert tuples to lists
            path_as_lists = [list(step) for step in trip.path]
            
            # --- EXPANSION STEP ---
            # Repeat the path sequence N times
            for _ in range(trip.times_taken):
                # We must append a COPY of the list structure to avoid mutability issues later when we adjust the deltas
                cities.extend(deepcopy(path_as_lists))

        previous = 0.0
        for city in cities:
            node_id = city[0]
            if node_id != 0:
                current_cumulative = city[1]
                city[1] = current_cumulative - previous
                
                # Sanity fix for float noise (e.g., -1e-16 becomes 0.0)
                if abs(city[1]) < 1e-9: 
                    city[1] = 0.0
                
                previous = current_cumulative
            else:
                # At base, we reset our 'previous' tracker because the truck is empty
                previous = 0.0
                # The delta at base is always 0 (we don't 'collect' at base)
                city[1] = 0.0

        return cities


def verify_delta_solution(cities_list, problem, tolerance=1e-3):
    """
    Verifies the solution AFTER it has been converted to 'delta' format.
    
    Format: [(node_id, gold_picked_up_here), ...]
    
    Checks:
    1. Connectivity: Do edges exist between sequential nodes?
    2. Gold Completeness: Does sum(gold_picked_up) for each city match the graph?
    3. Physics: Are we picking up negative gold? (Impossible)
    """
    
    # 1. Connectivity Check
    if not cities_list or cities_list[0][0] != 0:
        return False, "Path must start at base (0)."

    collected_gold = {c: 0.0 for c in range(problem.num_cities)}

    for i in range(len(cities_list)):
        current_node = cities_list[i][0]
        current_delta = cities_list[i][1]

        # A. Check Edge Existence (skip for first node)
        if i > 0:
            prev_node = cities_list[i-1][0]
            if prev_node != current_node:
                # Note: We use the private graph accessor or public property
                if not problem.graph.has_edge(prev_node, current_node):
                    return False, f"Step {i}: No edge between {prev_node} and {current_node}"

        # B. Check for Negative Gold (Bug in delta calculation)
        if current_delta < -1e-9:
             return False, f"Step {i} (City {current_node}): Negative gold collected ({current_delta})."

        # C. Accumulate Gold
        if current_node != 0:
            collected_gold[current_node] += current_delta

    # 2. Completeness Check (with tolerance)
    missing_log = []
    for c in range(1, problem.num_cities):
        total_available = problem.graph.nodes[c]['gold']
        total_collected = collected_gold[c]
        
        # Check if we missed gold (Undershoot)
        if total_available - total_collected > tolerance:
            missing_log.append(f"City {c}: Collected {total_collected:.2f} / {total_available:.2f}")
        
        # Check if we invented gold (Overshoot - usually a bug in accumulation)
        elif total_collected - total_available > tolerance:
             missing_log.append(f"City {c}: OVER-COLLECTED {total_collected:.2f} / {total_available:.2f}")

    if missing_log:
        return False, "Gold Validation Failed:\n" + "\n".join(missing_log)

    return True, "Valid"




def solve(problem: Problem) -> Solution:


    if problem.beta < 10.0:
        print("Running Parallel Genetic Algorithm with Migration...")
        archipelago = Archipelago(problem)
        solution = archipelago.run_parallel(total_generations=200, migration_interval=20)
    else:
        # If beta is very high, the cost is dominated by the gold term, so we can use a more aggressive heuristic that focuses on gold collection without worrying too much about distance. 
        # The Adaptive Split heuristic is designed for this kind of scenario, as it optimizes the number of visits to each city based on the gold available and the cost function's sensitivity to gold.
        # In general it makes no sese to run the genetic algorithm
        print("Running Heuristic Adaptive Split (No Parallelism)...")
        final_solution = generate_adaptive_split(problem, max_search=1000)
        cities = solution_to_cities(final_solution)
        is_valid, message = verify_delta_solution(cities, problem, tolerance=1e-3)
        if not is_valid:
            print(f"FINAL CHECK FAILED: {message}")
        return cities, final_solution.total_cost, is_valid, message
    
    
    return solution
