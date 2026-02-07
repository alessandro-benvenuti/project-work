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
from gold_collector.utils import generate_baseline, generate_topology_savings, generate_split_visits, generate_random_chunk_visits, generate_adaptive_split, compute_distance_matrix
from gold_collector.genetic import Island, run_island_evolution

class Archipelago:
    def __init__(self, problem, num_islands=3, population_size=50):
        self.problem = problem
        self.islands = []
        self.distance_matrix = compute_distance_matrix(problem)

        # Initialize islands with different strategies or seed solutions
        for i in range(num_islands):
            # Each island gets a different seed solution
            if i==0:
                seed_solution = generate_baseline(problem)
            elif i==1:
                seed_solution = generate_topology_savings(problem)
            else:
                seed_solution = generate_adaptive_split(problem)

            island = Island(
                island_id=i,
                strategy="random",
                seed_solution=seed_solution,
                problem=problem,
                dist_matrix=self.distance_matrix,
                population_size=population_size
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
        cost = best_solution.to_solution().total_cost

        cities = []
        for trip in best_solution.trips:
            cities.extend(trip.path)

        return cities, cost
        


def solve(problem: Problem) -> Solution:

    archipelago = Archipelago(problem)
    solution = archipelago.run_parallel(total_generations=200, migration_interval=20)
    
    
    return solution
