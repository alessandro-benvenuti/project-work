# gold_collector/genetic.py
import copy
import os
import os
import numpy as np
import random

from gold_collector.utils import generate_baseline, generate_topology_savings, generate_split_visits, generate_random_chunk_visits, generate_adaptive_split, compute_distance_matrix
from gold_collector.core import Solution, Trip

class Individual:
    def __init__(self, trips, cost):
        self.trips = trips
        self.cost = cost
        # We store cost immediately to avoid re-calc

    def to_solution(self, problem):
        # Converts the individual's trips into a Solution object, which can be evaluated or returned as output.
        # Note that we need to avoid appriximations here, so we will compute the exact path and cost for each trip.
        sol_trips = []
        for trip in self.trips:
            tcities = trip.cities
            # We need to compute the exact path for this trip using A* with the gold-aware
            exact_trip = Trip(
                cities=tcities, 
                problem=problem,
                path_matrix=None,         # Ensure no matrix is used
                use_precompute=False,     # Force EXACT mode (A*)
                use_dijkstra=True         # Force Dijkstra to ensure we get the true shortest path (no heuristic)
            )
            sol_trips.append(exact_trip)
        
        return Solution(sol_trips)

    def mutate(self, problem, path_matrix=None, use_precompute=False, strategy="random", num_mutations=1):
        # This function applies mutations to the individual's trips based on the specified strategy.
        s = strategy
        for _ in range(num_mutations):
            # Depending on the strategy, we can perform different types of mutations:
            if s == "random" or s not in ["merge", "split", "swap", "gold_split"]:
                strategy = random.choice(["merge", "split", "swap", "gold_split"])
            
            if strategy == "merge":
                # Merge two random trips into one (if possible)
                if len(self.trips) > 1:
                    t1, t2 = random.sample(self.trips, 2)

                    cities = []
                    for city in t1.cities:
                        cities.append(city)
                    for city in t2.cities:
                        # Avoid duplicating the tuple if it's exact same obj, 
                        # but we need to merge gold if city ID is same.
                        found = False
                        for i, existing in enumerate(cities):
                            if existing[0] == city[0]:
                                cities[i] = (existing[0], existing[1] + city[1])
                                found = True
                                break
                        if not found:
                            cities.append(city)

                    new_trip = Trip(cities, problem, path_matrix=path_matrix, use_precompute=use_precompute)

                    if new_trip:
                        self.cost = self.cost - t1.total_cost - t2.total_cost + new_trip.total_cost
                        self.trips.remove(t1)
                        self.trips.remove(t2)
                        self.trips.append(new_trip)

            elif strategy == "split":
                # Split a random trip into two (if possible)
                t = random.choice(self.trips)
                if len(t.cities) > 2: 
                    split_point = random.randint(1, len(t.cities)-1) # Fix index range
                    t1_cities = t.cities[:split_point]
                    t2_cities = t.cities[split_point:]
                    
                    # Create NEW trips
                    t1 = Trip(t1_cities, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                    t2 = Trip(t2_cities, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                    
                    self.cost = self.cost - t.total_cost + t1.total_cost + t2.total_cost
                    self.trips.remove(t)
                    self.trips.append(t1)
                    self.trips.append(t2)
                else:
                    # If we can't split, maybe try a different mutation
                    pass

            elif strategy == "swap":
                # Swap two cities between two random trips (if possible)
                if len(self.trips) > 1:
                    t1, t2 = random.sample(self.trips, 2)
                    
                    # Copy the city lists so we don't mutate the original Trip objects
                    c_list1 = list(t1.cities)
                    c_list2 = list(t2.cities)
                    
                    if c_list1 and c_list2:
                        # Perform the swap on the lists
                        idx1 = random.randrange(len(c_list1))
                        idx2 = random.randrange(len(c_list2))
                        
                        city1 = c_list1[idx1]
                        city2 = c_list2[idx2]
                        
                        c_list1[idx1] = city2
                        c_list2[idx2] = city1
                        
                        # Create new Trip objects to force path re-calculation
                        new_t1 = Trip(c_list1, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                        new_t2 = Trip(c_list2, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                        
                        # Update Cost and Population
                        self.cost = self.cost - t1.total_cost - t2.total_cost + new_t1.total_cost + new_t2.total_cost
                        self.trips.remove(t1)
                        self.trips.remove(t2)
                        self.trips.append(new_t1)
                        self.trips.append(new_t2)
                else:
                    # If we can't swap, maybe try a different mutation
                    pass
            
            elif strategy == "gold_split":
                # pick a random trip and for each city in the trip take just a portion of the gold, then split the trip into two
                t = random.choice(self.trips)
                if t.cities:
                    portion = random.uniform(0.1, 0.9)
                    new_cities = []
                    old_cities = []
                    for city in t.cities:
                        gold = city[1] * portion
                        new_cities.append((city[0], gold))
                        old_cities.append((city[0], city[1] - gold))
                    
                    t1 = Trip(new_cities, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                    t2 = Trip(old_cities, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                    
                    self.cost = self.cost - t.total_cost + t1.total_cost + t2.total_cost
                    self.trips.remove(t)
                    self.trips.append(t1)
                    self.trips.append(t2)
                
        return self
    
    def crossover(self, other_parent, problem, path_matrix=None, use_precompute=False):
        """
        Greedy Route Crossover (Inherit & Trim)
        1. Inherit random trips from Self (Parent A).
        2. Inherit trips from Other (Parent B), but trim already-collected gold.
        3. If gold is still missing, add simple recovery trips.
        """
        child_trips = []
        
        # Track gold collected by the Child so far
        # Initialize with 0.0 collected for all cities
        collected_gold = {c: 0.0 for c in range(problem.num_cities)}
        
        # --- STEP 1: Inherit from Parent A (Self) ---
        # Take random 50% of trips
        num_to_take = max(1, len(self.trips) // 2)
        trips_from_A = random.sample(self.trips, num_to_take)
        
        for t in trips_from_A:
            # Add trip to child
            child_trips.append(copy.deepcopy(t))
            
            # Update gold tracker
            for city_id, amount in t.cities:
                collected_gold[city_id] += amount

        # --- STEP 2: Inherit from Parent B (Other) ---
        # We try to add ALL trips from B, but we "clean" them first.
        
        for t in other_parent.trips:
            new_cities_for_trip = []
            
            for city_id, intended_amount in t.cities:
                total_available = problem.graph.nodes[city_id]['gold']
                already_collected = collected_gold.get(city_id, 0.0)
                remaining = total_available - already_collected
                
                # If there is gold left to collect
                if remaining > 1.0: # Tolerance for float noise
                    # We take the intended amount, OR whatever is left, whichever is smaller
                    amount_to_take = min(intended_amount, remaining)
                    
                    new_cities_for_trip.append((city_id, amount_to_take))
                    collected_gold[city_id] += amount_to_take
            
            # If the trimmed trip still has stops, add it
            if new_cities_for_trip:
                # We must create a new Trip object (runs A*)
                new_trip = Trip(new_cities_for_trip, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                if new_trip.total_cost != float('inf'):
                    child_trips.append(new_trip)

        # --- STEP 3: Repair (The "Dust" Collector) ---
        # Check if any city is missed entirely (rare but possible)
        missing_cities = []
        for c in range(1, problem.num_cities):
            total = problem.graph.nodes[c]['gold']
            current = collected_gold.get(c, 0.0)
            if total - current > 1.0:
                missing_cities.append((c, total - current))
        
        # If we missed anything, just add simple dedicated trips (Baseline style)
        # The mutation steps later will optimize these bad trips into good ones.
        for city_data in missing_cities:
            child_trips.append(Trip([city_data], problem, path_matrix=path_matrix, use_precompute=use_precompute))
            
        return Individual(child_trips, sum(t.total_cost for t in child_trips))


        

class Island:
    def __init__(self, island_id, strategy, seed_solution, problem, path_matrix=None, use_precompute=False, population_size=50, offspring_size=5):
        self.id = island_id
        self.strategy = strategy
        self.problem = problem
        self.path_matrix = path_matrix
        self.use_precompute = use_precompute
        self.population = [] # List of Individuals
        self.history = set() # Set of costs to prevent clones
        self.pop_size = population_size
        self.offspring_size = offspring_size

        # Initialize from seed
        initial_individual = Individual(seed_solution.trips, seed_solution.total_cost)
        self.population.append(initial_individual)
        self.history.add(initial_individual.cost)

    
    def tournament_select(self, k=3):
        """
        Selects the best individual from k random candidates.
        """
        # Fast sampling without replacement
        candidates = random.sample(self.population, k)
        # Return the one with the lowest cost
        return min(candidates, key=lambda ind: ind.cost)
            

    def process_generation(self, mutation_rate=0.2):
        # Select parents, mutate, replace weak...
        offspring = []

        # initialize population if empty (should only happen at the very beginning)
        if len(self.population) < self.pop_size:
            seed = self.population[0]

            while len(self.population) < self.pop_size:
                # Create mutated individuals from the seed solution, this ensures we start with a diverse population while still being grounded in a good solution.
                mutated_sol = copy.deepcopy(seed).mutate(self.problem, self.path_matrix, self.use_precompute, self.strategy, num_mutations=5)
                # Ensure uniqueness
                while mutated_sol.cost in self.history:
                    mutated_sol = copy.deepcopy(seed).mutate(self.problem, self.path_matrix, self.use_precompute, self.strategy, num_mutations=5)
                
                self.population.append(mutated_sol)
                self.history.add(mutated_sol.cost)

        for _ in range(self.offspring_size):
            if random.random() < mutation_rate:
                # Mutation step
                parent = self.tournament_select()
                mutated = copy.deepcopy(parent).mutate(self.problem, self.path_matrix, self.use_precompute, self.strategy, num_mutations=5)
                if mutated.cost not in self.history:
                    offspring.append(mutated)
                    self.history.add(mutated.cost)
            else:
                # Crossover step
                parent1 = self.tournament_select()
                parent2 = self.tournament_select()
                child = parent1.crossover(parent2, self.problem, path_matrix=self.path_matrix, use_precompute=self.use_precompute)
                if child.cost not in self.history:
                    offspring.append(child)
                    self.history.add(child.cost)

        # Replace worst individuals with new offspring
        self.population += offspring
        self.population.sort(key=lambda ind: ind.cost)
        self.population = self.population[:self.pop_size]
        
        

# --- THE WORKER FUNCTION ---
def run_island_evolution(island, generations):
    """
    This function runs on a separate CPU core.
    It takes an Island, runs N generations, and returns the modified Island.
    """

    pid = os.getpid()
    print(f"--> WORKER REPORTING: Island {island.id} is active on Process ID {pid}")

    for _ in range(generations):
        island.process_generation()
    return island