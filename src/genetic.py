# gold_collector/genetic.py
import copy
import os
import os
import numpy as np
import random

from src.core import Solution, Trip

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
                times_taken=trip.times_taken,  # Preserve the number of repeats
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
            if s == "random" or s not in ["merge", "split", "swap", "gold_split", "adjust_repeats"]:
                strategy = random.choice(["merge", "split", "swap", "gold_split", "adjust_repeats"])
            
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
                        self.cost = self.cost - t1.total_cost / t1.times_taken - t2.total_cost / t2.times_taken + new_trip.total_cost
                        if t1.times_taken > 1:
                            t1.change_times_taken(t1.times_taken - 1)
                        else:
                            self.trips.remove(t1)
                        if t2.times_taken > 1:
                            t2.change_times_taken(t2.times_taken - 1)
                        else:
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
                    
                    self.cost = self.cost - t.total_cost / t.times_taken + t1.total_cost + t2.total_cost
                    if t.times_taken > 1:
                        t.change_times_taken(t.times_taken - 1)
                    else:
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
                        self.cost = self.cost - t1.total_cost / t1.times_taken - t2.total_cost / t2.times_taken + new_t1.total_cost + new_t2.total_cost
                        if t1.times_taken > 1:
                            t1.change_times_taken(t1.times_taken - 1)
                        else:
                            self.trips.remove(t1)
                        if t2.times_taken > 1:
                            t2.change_times_taken(t2.times_taken - 1)
                        else:
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
                    
                    self.cost = self.cost - t.total_cost / t.times_taken + t1.total_cost + t2.total_cost
                    if t.times_taken > 1:
                        t.change_times_taken(t.times_taken - 1)
                    else:
                        self.trips.remove(t)
                    self.trips.append(t1)
                    self.trips.append(t2)

            elif strategy == "adjust_repeats":
                # Select a trip that acts as a "stack" of identical journeys
                candidates = [t for t in self.trips if t.times_taken > 0]
                if candidates:
                    t = random.choice(candidates)
                    
                    # 1. Calculate the TOTAL gold this stack is currently responsible for
                    current_stack_gold = sum(amount for _, amount in t.cities) * t.times_taken
                    
                    if current_stack_gold <= 0: continue

                    # 2. Determine new number of repeats and try to nudge it slightly
                    change = random.choice([-1, 1, -2, 2, -5, 5])
                    new_times = max(1, t.times_taken + change)
                    
                    if new_times == t.times_taken: continue

                    # 3. We keep the total gold responsibility the same, so we adjust the per-visit amount accordingly
                    new_cities = []
                    for city_id, old_amount in t.cities:
                        total_stack_gold = old_amount * t.times_taken
                        
                        # Ensure we don't accidentally create gold due to float drift
                        max_available = problem.graph.nodes[city_id]['gold']
                        if total_stack_gold > max_available:
                            total_stack_gold = max_available
                        
                        # Recalculate per-visit amount
                        new_amount = total_stack_gold / new_times
                        new_cities.append((city_id, new_amount))
                    
                    # 4. Create new Trip
                    new_trip = Trip(
                        cities=new_cities, 
                        problem=problem, 
                        path_matrix=path_matrix, 
                        use_precompute=use_precompute,
                        times_taken=new_times 
                    )
                    
                    self.cost = self.cost - t.total_cost + new_trip.total_cost
                    self.trips.remove(t)
                    self.trips.append(new_trip)
                
        return self
    
    def crossover(self, other_parent, problem, path_matrix=None, use_precompute=False):
        child_trips = []
        collected_gold = {c: 0.0 for c in range(problem.num_cities)}
        
        # 1. Inherit from Parent A
        num_to_take = max(1, len(self.trips) // 2)
        trips_from_A = random.sample(self.trips, num_to_take)
        
        for t in trips_from_A:
            new_t = copy.deepcopy(t)
            child_trips.append(new_t)
            for city_id, amount in new_t.cities:
                collected_gold[city_id] += amount * new_t.times_taken

        # 2. Inherit from Parent B
        for t in other_parent.trips:
            # Check if this trip is useful
            limit_factor = float('inf') 
            useful = False
            
            for city_id, amount_per_visit in t.cities:
                total_available = problem.graph.nodes[city_id]['gold']
                current_collected = collected_gold.get(city_id, 0.0)
                remaining = total_available - current_collected
                
                # If even a tiny bit of gold is left, this trip might be useful
                if remaining > 1e-6:
                    useful = True
                    # How many full repeats fit?
                    max_reps = int(remaining / amount_per_visit + 1e-9)
                    if max_reps < limit_factor:
                        limit_factor = max_reps
                else:
                    # City is full, we can't visit it with this trip configuration
                    limit_factor = 0
            
            if useful and limit_factor > 0:
                new_t = copy.deepcopy(t)
                new_t.change_times_taken(min(new_t.times_taken, limit_factor))
                
                child_trips.append(new_t)
                for city_id, amount in new_t.cities:
                    collected_gold[city_id] += amount * new_t.times_taken

        # 3. Repair the child by adding "sweeper trips" for any missing gold
        missing_cities = []
        for c in range(1, problem.num_cities):
            total = problem.graph.nodes[c]['gold']
            current = collected_gold.get(c, 0.0)
            
            if total - current > 1e-5:
                # We append the exact missing amount
                missing_cities.append((c, total - current))
        
        for city_data in missing_cities:
            # Create a dedicated "Sweeper Trip" for this dust
            child_trips.append(Trip(
                cities=[city_data], 
                problem=problem, 
                path_matrix=path_matrix, 
                use_precompute=use_precompute, 
                times_taken=1
            ))
            
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

        offspring = []

        # Initialize population if empty (should only happen at the very beginning)
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

        # Generate offspring through mutation and crossover
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