import random
import numpy as np
import networkx as nx

from gold_collector.core import Solution, Trip

def get_euclidean_distance(node_a, node_b, graph):
    """
    Fastest possible distance check.
    Uses raw coordinates from the graph metadata, bypassing NetworkX overhead.
    """
    pos_a = np.array(graph.nodes[node_a]['pos'])
    pos_b = np.array(graph.nodes[node_b]['pos'])
    return np.linalg.norm(pos_a - pos_b)

def generate_fast_baseline(problem):
    """
    Super-fast Baseline Generator.
    Runs Dijkstra ONCE to solve all outbound paths simultaneously.
    Assumes Return Path = Reverse(Outbound Path).
    """
    
    # Run Dijkstra ONCE from base (0), this gives us the path to EVERY node in the graph.
    # We use weight='dist' because outbound trucks are empty (cost = distance).
    all_outbound_paths = nx.single_source_dijkstra_path(problem.graph, 0, weight='dist')
    
    trips = []
    
    for city in range(1, problem.num_cities):
        if city not in all_outbound_paths:
            continue
            
        # Get the path: [0, node_a, node_b, ..., city]
        outbound_nodes = all_outbound_paths[city]
        
        # Heuristic: Assume return path is the exact reverse
        # [city, ..., node_b, node_a, 0]
        return_nodes = outbound_nodes[::-1] 
        
        # Build the full path list for the Trip object
        # Format: [(node_id, gold_carried), ...]
        full_path = []
        city_gold = problem.graph.nodes[city]['gold']
        
        # Add Outbound (Gold = 0)
        # We skip the last node (city) to handle it in the transition
        for node in outbound_nodes[:-1]:
            full_path.append((node, 0.0))
            
        # Add Return (Gold = city_gold)
        for node in return_nodes:
            full_path.append((node, city_gold))
            
        # Create Trip using the precomputed path
        # This skips the internal A* call, making it instant.

        trips.append(Trip(
            cities=[(city, city_gold)], 
            problem=problem, 
            precomputed_path=full_path
        ))
        
    return Solution(trips)

def generate_baseline(problem):
    """Heuristic 1: One city per trip --> same idea as the baseline solution"""
    trips = []
    for city in range(1, problem.num_cities):
        trips.append(Trip([(city, problem.graph.nodes[city]['gold'])], problem))
    return Solution(trips)

def generate_smart_baseline(problem: Problem):
    if problem._beta <= 1:
        return generate_fast_baseline(problem)
    else:
        return generate_baseline(problem)

def generate_topology_savings(problem):
    """
    Heuristic 3: 'Path Interception'
    
    Strategy:
    Merge Trip A and Trip B if Trip B's first city where we are picking up gold (or its neighbor)
    is physically traversed by Trip A on its way home.
    """
    
    # 1. Initialize fast baseline
    base = generate_fast_baseline(problem)
    trips = base.trips.copy()
    random.shuffle(trips)
    
    improved = True
    while improved:
        improved = False
        current_trips = list(trips)     # Snapshot to iterate over
        
        # Build a quick lookup map: Start City -> Trip Object
        # This lets us instantly find 'Trip B' if we hit 'City B' on our path
        start_node_to_trip = {t.cities[0][0]: t for t in trips}
        
        for trip_a in current_trips:
            if trip_a not in trips: continue 

            # Extract the return path nodes from the trip
            # (Assuming trip.path is a list of tuples like [(node, gold), ...])
            # We only care about the nodes visited AFTER the last collection
            
            # Find index of last collection in the path
            last_collection_city = trip_a.cities[-1][0]
            
            # Scan the path segments
            return_path_nodes = set()
            collecting_phase_ended = False
            
            for node, _ in reversed(trip_a.path):
                if node == last_collection_city:
                    break # Stop! We have reached the collection point.
                
                if node != 0: # Ignore base
                    return_path_nodes.add(node)
            
            best_merge = None
            max_gain = 0
            
            # NOW CHECK: Does this path hit the start of any other trip?
            best_merge = None
            max_gain = 0
            
            # Check for direct interception
            for other_start_node, trip_b in start_node_to_trip.items():
                if trip_b is trip_a: continue
                
                # Do we traverse B OR a neighbor of B?
                
                is_intercepted = False
                if other_start_node in return_path_nodes:
                    is_intercepted = True
                else:
                    # Optional: Check neighbors for wider catchment
                    # neighbors = list(problem.graph.neighbors(other_start_node))
                    # if any(n in return_path_nodes for n in neighbors):
                    #     is_intercepted = True
                    pass

                if is_intercepted:
                    # Valid Candidate Found! 
                    # Now we check the cost (Accurate Check)
                    combined_cities = trip_a.cities + trip_b.cities
                    
                    # Estimate gain
                    # We can afford to run Trip() here because interception is rare/specific
                    new_trip = Trip(combined_cities, problem)
                    gain = (trip_a.total_cost + trip_b.total_cost) - new_trip.total_cost
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_merge = (trip_b, new_trip)
            
            # Apply merge
            if best_merge:
                trip_b, new_trip = best_merge
                
                # Update our lookup map and list
                del start_node_to_trip[trip_b.cities[0][0]]
                del start_node_to_trip[trip_a.cities[0][0]]
                
                trips.remove(trip_a)
                trips.remove(trip_b)
                trips.append(new_trip)
                
                # Add new trip to lookup for future merges
                start_node_to_trip[new_trip.cities[0][0]] = new_trip
                
                improved = True
                break 

    return Solution(trips)

def generate_geometric_clusters(problem):
    """Heuristic 3: K-Means or Angular Sweep"""
    pass

def generate_split_visits(problem):
    """Heuristic 4: Visit every city twice (50% gold each)"""
    trips = []
    for city in range(1, problem.num_cities):
        half_gold = problem.graph.nodes[city]['gold'] / 2
        trips.append(Trip([(city, half_gold)]))
        trips.append(Trip([(city, half_gold)]))
    return Solution(trips)