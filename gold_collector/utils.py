import random
import numpy as np
import networkx as nx
from copy import copy, deepcopy

from gold_collector.core import Solution, Trip

def get_euclidean_distance(node_a, node_b, graph):
    """
    Fastest possible distance check.
    Uses raw coordinates from the graph metadata, bypassing NetworkX overhead.
    """
    pos_a = np.array(graph.nodes[node_a]['pos'])
    pos_b = np.array(graph.nodes[node_b]['pos'])
    return np.linalg.norm(pos_a - pos_b)

def generate_baseline(problem):
    """
    Smart Baseline Generator.
    1. Runs Dijkstra ONCE to get all outbound paths (Distance Optimal).
    2. Feeds these paths to Trip as 'static_paths'.
    3. The Trip class automatically decides:
       - Use outbound path? YES (Always optimal for empty truck).
       - Use inbound path? Only if Beta <= 1. Otherwise, run A*.
    """
    
    # Use _graph to ensure O(1) access speed (avoid copying)
    G = problem._graph 
    
    # 1. Run Dijkstra ONCE from base (0)
    # This gives us the optimal distance path to EVERY city.
    all_outbound_paths = nx.single_source_dijkstra_path(G, 0, weight='dist')
    
    trips = []
    
    for city in range(1, problem.num_cities):
        if city not in all_outbound_paths:
            continue 
            
        outbound_nodes = all_outbound_paths[city]
        
        # Prepare the segments
        # Outbound: 0 -> ... -> City
        # Inbound Heuristic: City -> ... -> 0 (Reverse of outbound)
        static_segments = {
            'outbound': outbound_nodes,
            'inbound': outbound_nodes[::-1]
        }
        
        # Create Trip
        # We DO NOT use 'precomputed_path' anymore.
        # We use 'static_paths' to let Trip apply its Hybrid Logic.
        trips.append(Trip(
            cities=[(city, G.nodes[city]['gold'])], 
            problem=problem, 
            static_paths=static_segments
        ))
        
    return Solution(trips)

def generate_topology_savings(problem, check_neighbors=False, sample_ratio=1):
    """
    Heuristic 3: 'Path Interception' (Optimized)
    
    Optimizations:
    1. Inverted Indexing: We look up path nodes in a dict (O(1)) instead of looping trips.
    2. Sampling: Only check 'sample_ratio' percent of trips each pass to save time.
    3. Neighbor Check: Uses graph topology to find near-misses.
    """
    
    # 1. Initialize fast baseline
    base = generate_baseline(problem)
    trips = base.trips.copy()
    
    # Use the graph for neighbor lookups
    G = problem._graph
    
    improved = True
    while improved:
        improved = False
        
        # 1. Build Fast Lookup: Start City -> Trip Object
        # This allows O(1) checking if a node is a start point for someone else
        start_node_to_trip = {t.cities[0][0]: t for t in trips}
        
        # 2. Create a candidate list (Sampling)
        # We shuffle and only look at the top X% to speed up large graphs
        candidates = list(trips)
        random.shuffle(candidates)
        
        limit = max(1, int(len(candidates) * sample_ratio))
        candidates = candidates[:limit]

        for trip_a in candidates:
            if trip_a not in trips: continue # Might have been merged in previous iter

            # --- Extract Return Path Nodes ---
            # We only care about nodes visited AFTER the last collection
            last_collection_city = trip_a.cities[-1][0]
            
            path_nodes_to_check = []
            collection_phase_ended = False
            
            # Walk backwards from the end (Base)
            for node, _ in reversed(trip_a.path):
                if node == last_collection_city:
                    break 
                if node != 0: # Ignore base
                    path_nodes_to_check.append(node)

            best_merge = None
            max_gain = 0
            
            # --- The Optimized Check ---
            # Instead of looping all 'trip_b's, we loop the nodes in our path
            # and ask: "Does anyone start here?"
            
            # We use a set to avoid checking the same node twice if the path loops
            nodes_visited = set()

            for path_node in path_nodes_to_check:
                if path_node in nodes_visited: continue
                nodes_visited.add(path_node)

                # List of potential start nodes to check (Current Node + Neighbors)
                potential_starts = [path_node]
                
                if check_neighbors:
                    # Add all neighbors of the current path node
                    potential_starts.extend(list(G.neighbors(path_node)))
                
                for target_node in potential_starts:
                    # FAST CHECK: Is there a trip starting at this target_node?
                    if target_node in start_node_to_trip:
                        trip_b = start_node_to_trip[target_node]
                        
                        if trip_b is trip_a: continue
                        
                        # --- Evaluation ---
                        # We found a candidate! Calculate cost.
                        combined_cities = trip_a.cities + trip_b.cities
                        new_trip = Trip(combined_cities, problem)
                        
                        gain = (trip_a.total_cost + trip_b.total_cost) - new_trip.total_cost
                        
                        if gain > max_gain:
                            max_gain = gain
                            best_merge = (trip_b, new_trip)

            # Apply merge if found
            if best_merge:
                trip_b, new_trip = best_merge
                
                # Remove old trips
                if trip_a in trips: trips.remove(trip_a)
                if trip_b in trips: trips.remove(trip_b)
                
                # Add new trip
                trips.append(new_trip)
                
                # Update lookup for subsequent iterations inside this loop? 
                # No, simpler to just break and restart the 'while' loop to rebuild indices cleanly.
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

        if half_gold == 0:
            continue

        trip_one = Trip([(city, half_gold)], problem=problem)
        trip_two = copy(trip_one)

        trips.append(trip_one)
        trips.append(trip_two)

    return Solution(trips)

def generate_random_chunk_visits(problem, max_split=4):
    """
    Heuristic 5: Random Chunk Visits

    For each city, randomly decide to split the collection into 'k' separate trips.
    k is chosen randomly between 1 (no split) and max_split.
    """
    
    # Splitting is only beneficial if Beta > 1. Otherwise just return the standard baseline (1 trip per city)
    if problem.beta <= 1:
        return generate_baseline(problem)

    trips = []
    
    for city in range(1, problem.num_cities):
        total_gold = problem.graph.nodes[city]['gold']
        if total_gold == 0: continue
        
        # Randomly decide how many chunks to split this city into
        num_visits = random.randint(1, max_split)
        
        gold_per_visit = total_gold / num_visits
        
        # We calculate the path ONCE.
        template_trip = Trip(
            cities=[(city, gold_per_visit)], 
            problem=problem
        )
        
        # 4. Clone the template 'num_visits' times
        for _ in range(num_visits):
            # We must Deep Copy to ensure they are independent objects
            # (e.g., if a mutation later changes one trip, it shouldn't change the others)
            trips.append(deepcopy(template_trip))
            
    return Solution(trips)

def generate_adaptive_split(problem, max_search=6):
    """
    Heuristic 6: Adaptive Optimal Split
    
    Strategy:
    For each city, numerically test k=1, k=2, ... k=max_search visits.
    Pick the 'k' that results in the lowest total cost for that specific city.
    
    Why:
    - Some far cities with little gold are best visited ONCE (k=1).
    - Nearby cities with massive gold might need THREE visits (k=3).
    - Removes the guesswork of 'Split Visits' and 'Random'.
    """
    
    # If Beta is small, don't waste time; 1 trip is always best.
    if problem.beta <= 1:
        return generate_baseline(problem)

    best_trips = []
    
    
    for city in range(1, problem.num_cities):
        total_gold = problem.graph.nodes[city]['gold']
        if total_gold == 0: continue
        
        
        # --- The Competition ---
        best_k_cost = float('inf')
        best_k_trips = []
        
        # Test k=1 to k=max_search
        for k in range(1, max_search + 1):
            gold_per_visit = total_gold / k
            
            # Create a TEMPORARY trip just to check its cost
            # We use precomputed nodes for speed
            temp_trip = Trip(
                cities=[(city, gold_per_visit)], 
                problem=problem
            )
            
            # Total cost for this strategy = Cost of ONE trip * k
            current_strategy_cost = temp_trip.total_cost * k
            
            if current_strategy_cost < best_k_cost:
                best_k_cost = current_strategy_cost
                # Store the winner. We need 'k' copies of this trip.
                # We store the object itself to clone later
                best_k_trips = [temp_trip] + [deepcopy(temp_trip) for _ in range(k-1)]
            else:
                # Since cost is unlikely to improve with higher k, we can break early
                break
        
        # Append the winning strategy for this city to the main list
        best_trips.extend(best_k_trips)
        
    return Solution(best_trips)