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

from src.core import Solution, Trip
from src.utils import generate_baseline, generate_topology_savings, generate_adaptive_split, compute_distance_matrix, precompute_weighted_paths
from src.genetic import Island, run_island_evolution

class Archipelago:
    def __init__(self, problem, num_islands=4, population_size=50, offspring_size=20):
        self.problem = problem
        self.islands = []
        self.distance_matrix = compute_distance_matrix(problem)

        # Determine if we should use the heuristic precomputation based on problem size and beta.
        # We choose num_cities > 200 as a heuristic cutoff for when to use the approximation.
        # Also if beta <= 1 then the cost is more distance-sensitive, so we can afford to precompute paths in any case to speeed up the search.
        self.use_precompute = (problem.beta <= 1.0) or (problem.num_cities > 200)
        
        if self.use_precompute:
            print("--- Precomputing Paths (Heuristic Mode) ---")
            self.path_matrix = precompute_weighted_paths(problem, ref_gold_ratio=0.5)
        else:
            self.path_matrix = None

        if self.problem.num_cities > 50:
            strategies = ["random", "merge", "adjust_repeats", "merge"]
        else:
            # Since we have few cities we cannot ensure unique solutions by forcing merge, so we use more randomization to ensure diversity in the population
            strategies = ["random", "random", "random", "random"]

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
        # Harvest the Elites
        migrants = [deepcopy(island.population[0]) for island in self.islands]

        # Inject into Neighbors
        for i in range(len(self.islands)):
            target_idx = (i + 1) % len(self.islands)
            target_island = self.islands[target_idx]
            migrant = migrants[i]

            # Immigration Control Check: Prevent duplicates in the target island's current population
            current_costs = {ind.cost for ind in target_island.population}
            if migrant.cost in current_costs:
                # The island already has this solution.
                continue

            # Replace the worst
            target_island.population[-1] = migrant
            
            # Update the target's history too, so it knows about this cost
            target_island.history.add(migrant.cost)

            # Re-sort
            target_island.population.sort(key=lambda x: x.cost)

        # Log the migration event
        best_costs = [isl.population[0].cost for isl in self.islands]
        print(f"--- Migration Complete. Island Bests: {['{:.2f}'.format(c) for c in best_costs]} ---")



    def run_parallel(self, total_generations=200, migration_interval=20):
        # We run in blocks (epochs)
        num_epochs = total_generations // migration_interval
        
        with ProcessPoolExecutor(max_workers=len(self.islands)) as executor:
            for epoch in range(num_epochs):
                print(f"--- Epoch {epoch+1}/{num_epochs} (Running parallel) ---")
                
                # 1. Send islands to workers
                # executor.map runs 'run_island_evolution' on each island in parallel
                futures = [executor.submit(run_island_evolution, island, migration_interval) 
                           for island in self.islands]
                
                # 2. Collect results back to main process
                self.islands = [f.result() for f in futures]
                
                # 3. Migration
                self.perform_migration()

                # 4. report global best after migration
                best_ind = min((isl.population[0] for isl in self.islands), key=lambda x: x.cost)
                print(f"Epoch {epoch+1}/{num_epochs} - Global Best: {best_ind.cost:.2f}")
                
                

        best_ones = [isl.population[0] for isl in self.islands]
        best_solution = min(best_ones, key=lambda ind: ind.cost)
        final_solution = best_solution.to_solution(self.problem)

        # Extract the full path from the solution
        cities = solution_to_cities(final_solution, self.problem)

        # Now we check if this final list is actually valid
        is_valid, message = verify_delta_solution(cities, self.problem, tolerance=1e-3)

        if not is_valid:
            print(f"FINAL CHECK FAILED: {message}")
        else:
            print(f"FINAL CHECK PASSED")

        return cities, final_solution.total_cost, is_valid, message
        

def solution_to_cities(solution: Solution, problem: Problem):
    """
    Converts a Solution object into a List of Tuples: [(node_id, gold_collected), ...]
    1. Calculates Deltas (incremental gold).
    2. Clamps values to physics (0 <= collected <= available).
    3. Removes the starting (0,0).
    4. Removes consecutive (0,0) duplicates.
    5. Returns a list of IMMUTABLE TUPLES.
    """
    # 1. Extract path as mutable LISTS first (so we can update gold values)
    cities = []
    for trip in solution.trips:
        # We assume trip.path is stored as tuples, so we convert to list to allow editing
        path_as_lists = [list(step) for step in trip.path]
        for _ in range(trip.times_taken):
            cities.extend(deepcopy(path_as_lists))

    # 2. Calculate Deltas and Clamp (In-Place modification of lists)
    collected_so_far = {c: 0.0 for c in range(problem.num_cities)}
    previous = 0.0
    
    for city in cities:
        node_id = city[0]
        if node_id != 0:
            raw_cumulative = city[1]
            delta = raw_cumulative - previous
            
            # Clamping logic
            if delta > 1e-9:
                max_val = problem.graph.nodes[node_id]['gold']
                current_owned = collected_so_far[node_id]
                allowed = max_val - current_owned
                
                if delta > allowed:
                    delta = max(0.0, allowed)
                
                collected_so_far[node_id] += delta
            else:
                delta = 0.0
            
            city[1] = delta
            previous = raw_cumulative
        else:
            previous = 0.0
            city[1] = 0.0

    # 3. FILTERING & CONVERSION TO TUPLES
    final_cities = []
    
    # A. Remove the very first node if it is (0,0)
    if cities and cities[0][0] == 0 and cities[0][1] == 0:
        cities.pop(0)

    if cities:
        # Add the first element as a TUPLE
        final_cities.append(tuple(cities[0]))
        
        # Loop through the rest to filter duplicates
        for i in range(1, len(cities)):
            current_node = cities[i]
            prev_node = cities[i-1]
            
            # Check for consecutive base visits
            is_zero = (current_node[0] == 0 and current_node[1] == 0)
            was_zero = (prev_node[0] == 0 and prev_node[1] == 0)
            
            if is_zero and was_zero:
                continue # Skip duplicate
            
            # B. CONVERT TO TUPLE HERE
            final_cities.append(tuple(current_node))

    return final_cities

def verify_delta_solution(cities_list, problem, tolerance=1e-3):
    """
    Verifies the solution AFTER it has been converted to 'delta' format.
    
    Format: [(node_id, gold_picked_up_here), ...]
    
    Checks:
    1. Connectivity: Do edges exist between sequential nodes?
    2. Gold Completeness: Does sum(gold_picked_up) for each city match the graph?
    3. Physics: Are we picking up negative gold? (Impossible)
    """
    
    # 1. Connectivity Check and Physics Check (while accumulating gold)
    collected_gold = {c: 0.0 for c in range(problem.num_cities)}

    for i in range(len(cities_list)):
        current_node = cities_list[i][0]
        current_delta = cities_list[i][1]

        # A. Check Edge Existence (skip for first node)
        if i > 0:
            prev_node = cities_list[i-1][0]
            if prev_node != current_node:
                if not problem.graph.has_edge(prev_node, current_node):
                    return False, f"Step {i}: No edge between {prev_node} and {current_node}"

        # B. Check for Negative Gold (for any bugs)
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
    '''
    Depending on the value of beta, we choose different strategies:
        If beta is low, the cost is more distance-sensitive, so we can afford to run a more complex genetic algorithm that optimizes both distance and gold collection.
        The algorithm works by having 4 different islands:
            Island 0 starts with the baseline solution and uses random mutations to explore the neighborhood.
            Island 1 starts with a topology-based savings solution and focuses on merging trips to reduce distance.
            Island 2 starts with an adaptive split solution and focuses on adjusting the number of repetitions to optimize the gold-distance tradeoff.
            Island 3 starts with a random solution and uses random mutations to explore more diverse solutions (this is a more exploratory island that can discover novel solutions that the heuristics might miss).
        The islands evolve in parallel and exchange their best solutions every 20 generations, allowing good solutions to propagate while maintaining diversity.

        If beta is high, the cost is more gold-sensitive, so we can use a more aggressive heuristic that focuses on gold collection without worrying too much about distance. 
        The Adaptive Split heuristic is designed for this kind of scenario, as it optimizes the number 
        of visits to each city based on the gold available and the cost function's sensitivity to gold.
        So we can say that running the genetic algorithm is not worth it in this case, since the search 
        space is more dominated by the gold collection aspect and we can get good results with a well-designed 
        heuristic that directly optimizes for that.

    '''


    # if problem.beta <= 1.5 or (problem.num_cities > 1000 and problem.beta > 1):
    if problem.num_cities < 1000 and problem.beta <= 1.5:
        print("Running Parallel Genetic Algorithm with Migration...")
        if problem.num_cities > 50:
            archipelago = Archipelago(problem)
        else:
            # For smaller problems we use just 3 islands, since the last 2 would otherwise be too similar
            archipelago = Archipelago(problem, num_islands=3)
        
        if problem.num_cities > 1000:
            path_matrix = precompute_weighted_paths(problem, ref_gold_ratio=0.5)
            solution = generate_topology_savings(problem, path_matrix=path_matrix, use_precompute=True)
        
        else:
            # run the genetic algorithm
            solution = archipelago.run_parallel(total_generations=200, migration_interval=20)
    else:
        if problem.beta > 1:
            # If beta is very high, the cost is dominated by the gold term, so we can use a more aggressive heuristic that focuses on gold collection without worrying too much about distance. 
            # The Adaptive Split heuristic is designed for this kind of scenario, as it optimizes the number of visits to each city based on the gold available and the cost function's sensitivity to gold.
            # In general it makes no sense to run the genetic algorithm
            print("Running Heuristic Adaptive Split (No Parallelism)...")
            final_solution = generate_adaptive_split(problem, max_search=1000)
            cities = solution_to_cities(final_solution, problem)
            is_valid, message = verify_delta_solution(cities, problem, tolerance=1e-3)
            if not is_valid:
                print(f"FINAL CHECK FAILED: {message}")
            return cities, final_solution.total_cost, is_valid, message
        
        else:
            print("Running Heuristic Topology Savings (No Parallelism)...")
            path_matrix = precompute_weighted_paths(problem, ref_gold_ratio=0.5)
            final_solution = generate_topology_savings(problem, path_matrix=path_matrix, use_precompute=True)
            cities = solution_to_cities(final_solution, problem)
            is_valid, message = verify_delta_solution(cities, problem, tolerance=1e-3)
            if not is_valid:
                print(f"FINAL CHECK FAILED: {message}")
            return cities, final_solution.total_cost, is_valid, message

    
    
    return solution
