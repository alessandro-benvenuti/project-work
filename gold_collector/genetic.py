# gold_collector/genetic.py
import copy
import numpy as np
import random

from gold_collector.utils import generate_baseline, generate_topology_savings, generate_split_visits, generate_random_chunk_visits, generate_adaptive_split, compute_distance_matrix
from gold_collector.core import Solution, Trip

class Individual:
    def __init__(self, trips, cost):
        self.trips = trips
        self.cost = cost
        # We store cost immediately to avoid re-calc

    def mutate(self, problem, dist_matrix, strategy, num_mutations=1):
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

                    # Create new trip with combined cities
                    cities = []
                    for city in t1.cities:
                        cities.append(city)
                    for city in t2.cities:
                        if city not in cities:
                            cities.append(city)
                        else:
                            # If city already in trip, sum the gold
                            idx = cities.index(city)
                            cities[idx] = (city[0], cities[idx][1] + city[1])

                    new_trip = Trip(cities, problem)
                    new_trip.compute_cost(problem)
                    if new_trip:
                        self.trips.remove(t1)
                        self.trips.remove(t2)
                        self.trips.append(new_trip)
                        self.cost = self.cost - t1.total_cost - t2.total_cost + new_trip.total_cost

            elif strategy == "split":
                # Split a random trip into two (if possible)
                t = random.choice(self.trips)
                if len(t.cities) > 2: # Need at least 3 cities to split
                    split_point = random.randint(1, len(t.cities)-2)
                    t1_cities = t.cities[:split_point]
                    t2_cities = t.cities[split_point:]
                    t1 = Trip(t1_cities, problem)
                    t2 = Trip(t2_cities, problem)
                    t1.compute_cost(problem)
                    t2.compute_cost(problem)
                    if t1 and t2:
                        self.trips.remove(t)
                        self.trips.append(t1)
                        self.trips.append(t2)
                        self.cost = self.cost - t.total_cost + t1.total_cost + t2.total_cost
                else:
                    # If we can't split, maybe try a different mutation
                    pass

            elif strategy == "swap":
                # Swap two cities between two random trips (if possible)
                if len(self.trips) > 1:
                    t1, t2 = random.sample(self.trips, 2)
                    if t1.cities and t2.cities:
                        c1 = random.choice(t1.cities)
                        c2 = random.choice(t2.cities)
                        t1_cost = t1.total_cost
                        t2_cost = t2.total_cost
                        t1.cities.remove(c1)
                        t2.cities.remove(c2)
                        t1.cities.append(c2)
                        t2.cities.append(c1)
                        t1.compute_cost(problem)
                        t2.compute_cost(problem)
                        self.cost = self.cost - t1_cost - t2_cost + t1.total_cost + t2.total_cost
                else:
                    # If we can't swap, maybe try a different mutation
                    pass
            
            elif strategy == "gold_split":
                # pick a random trip and for each city in the trip take just a portion of the gold, then split the trip into two
                t = random.choice(self.trips)
                portion = random.uniform(0.1, 0.9)
                new_cities = []
                old_cities = []
                for city in t.cities:
                    gold = city[1] * portion
                    new_cities.append((city[0], gold))
                    old_cities.append((city[0], city[1] - gold))
                t1 = Trip(new_cities, problem)
                t2 = Trip(old_cities, problem)
                t1.compute_cost(problem)
                t2.compute_cost(problem)
                if t1 and t2:
                    self.trips.remove(t)
                    self.trips.append(t1)
                    self.trips.append(t2)
                    self.cost = self.cost - t.total_cost + t1.total_cost + t2.total_cost
                
        return self


        

class Island:
    def __init__(self, island_id, strategy, seed_solution, problem, dist_matrix, population_size=50):
        self.id = island_id
        self.strategy = strategy
        self.problem = problem
        self.dist_matrix = dist_matrix
        self.population = [] # List of Individuals
        self.history = set() # Set of costs to prevent clones
        
        # Initialize from seed
        initial_individual = Individual(seed_solution.trips, seed_solution.total_cost)
        self.population.append(initial_individual)
        self.history.add(initial_individual.cost)

        for _ in range(population_size-1):
            # Create mutated individuals from the seed solution, this ensures we start with a diverse population while still being grounded in a good solution.
            mutated_sol = copy.deepcopy(initial_individual).mutate(problem, dist_matrix, strategy, num_mutations=5)
            # Ensure uniqueness
            while mutated_sol.cost in self.history:
                mutated_sol = copy.deepcopy(initial_individual).mutate(problem, dist_matrix, strategy, num_mutations=5)
            
            self.population.append(mutated_sol)
            self.history.add(mutated_sol.cost)
            

    def process_generation(self):
        # Select parents, mutate, replace weak...
        pass

# --- THE WORKER FUNCTION ---
def run_island_evolution(island, generations):
    """
    This function runs on a separate CPU core.
    It takes an Island, runs N generations, and returns the modified Island.
    """
    for _ in range(generations):
        island.process_generation()
    return island