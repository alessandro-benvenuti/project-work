import random
import numpy as np
import networkx as nx
from copy import copy, deepcopy

from gold_collector.core import Solution, Trip

def precompute_weighted_paths(problem, ref_gold_ratio=0.5):
    """
    Generates a dictionary of optimal paths between all pairs of nodes.
    The optimization assumes the truck carries 'ref_gold_ratio' * AvgGold.
    
    This creates 'Fixed Geometry' that respects the Beta > 1 penalty 
    (preferring short hops) without running A* every time.
    """
    # 1. Calculate Reference Gold (The "Average" Load)
    total_gold_in_world = sum(node[1]['gold'] for node in problem.graph.nodes(data=True))
    # Average gold per city? Or average load? 
    # A safe heuristic is: The truck is usually half full relative to average city gold
    avg_city_gold = total_gold_in_world / problem.num_cities
    
    # Heuristic: We design paths assuming we carry this much gold
    ref_gold = avg_city_gold * ref_gold_ratio
    
    # 2. Create a temporary graph with STATIC weights
    # We bake the cost formula into the edge weight
    G_weighted = problem.graph.copy()
    
    for u, v, data in G_weighted.edges(data=True):
        d = data['dist']
        # The Cost Formula: dist + (alpha * dist * ref_gold)^beta
        weight = d + (problem.alpha * d * ref_gold) ** problem.beta
        data['weight'] = weight

    # 3. Run All-Pairs Dijkstra ONCE
    # This returns a generator, convert to dict
    # format: paths[source][target] = [list of nodes]
    # Memory: ~200MB for 1000 cities. fast and safe.
    paths = dict(nx.all_pairs_dijkstra_path(G_weighted, weight='weight'))
    
    return paths

def get_euclidean_distance(node_a, node_b, graph):
    """
    Fastest possible distance check.
    Uses raw coordinates from the graph metadata, bypassing NetworkX overhead.
    """
    pos_a = np.array(graph.nodes[node_a]['pos'])
    pos_b = np.array(graph.nodes[node_b]['pos'])
    return np.linalg.norm(pos_a - pos_b)

def compute_distance_matrix(problem):
    """
    Creates a 2D array where matrix[i][j] is the TRUE graph distance
    between city i and city j.
    """
    # Get all shortest path lengths using Dijkstra (handles weighted edges correctly)
    # This returns an iterator, so we convert to dict
    length_iter = nx.all_pairs_dijkstra_path_length(problem.graph, weight='dist')
    dist_dict = dict(length_iter)
    
    num_nodes = problem.num_cities
    matrix = np.zeros((num_nodes, num_nodes))
    
    for u in range(num_nodes):
        for v in range(num_nodes):
            # If path exists, store it. If isolated (unlikely in connected), use infinity.
            if v in dist_dict[u]:
                matrix[u][v] = dist_dict[u][v]
            else:
                matrix[u][v] = float('inf')
                
    return matrix

def generate_baseline(problem, path_matrix=None, use_precompute=False):
    """
    Smart Baseline Generator.
    Creates one trip per city.
    Uses path_matrix if available for O(1) path retrieval.
    Otherwise falls back to A* (via the Trip class).
    """
    trips = []

    # Iterate 1 to N (skipping base 0)
    for city in range(1, problem.num_cities):
        
        # Optimization: Don't visit cities with no gold
        gold_amount = problem.graph.nodes[city]['gold']
        if gold_amount <= 0:
            continue
            
        # Create a dedicated trip for this city
        # The Trip class handles the routing logic (Approx vs Exact) internally
        new_trip = Trip(
            cities=[(city, gold_amount)], 
            problem=problem, 
            path_matrix=path_matrix, 
            use_precompute=use_precompute
        )
        
        trips.append(new_trip)
    
    return Solution(trips)

def generate_topology_savings(problem, check_neighbors=False, sample_ratio=1, path_matrix=None, use_precompute=False):
    """
    Heuristic 3: 'Path Interception' (Optimized)
    
    Optimizations:
    1. Inverted Indexing: We look up path nodes in a dict (O(1)) instead of looping trips.
    2. Sampling: Only check 'sample_ratio' percent of trips each pass to save time.
    3. Neighbor Check: Uses graph topology to find near-misses.
    """
    
    # 1. Initialize fast baseline
    base = generate_baseline(problem, path_matrix=path_matrix, use_precompute=use_precompute)
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
                        new_trip = Trip(combined_cities, problem, path_matrix=path_matrix, use_precompute=use_precompute)
                        
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

def generate_adaptive_split(problem, max_search=1000, path_matrix=None, use_precompute=False):
    """
    Heuristic 6: Adaptive Optimal Split (Binary Search Optimized)
    
    Uses Binary Search to find the optimal 'k' visits minimizing total cost.
    Complexity: O(log max_search) instead of O(max_search).
    """
    
    # If Beta <= 1, splitting doesn't help. Just return the standard baseline.
    if problem.beta <= 1:
        return generate_baseline(problem, path_matrix=path_matrix, use_precompute=use_precompute)

    best_trips = []
    
    for city in range(1, problem.num_cities):
        total_gold = problem.graph.nodes[city]['gold']
        if total_gold == 0: continue
        
        # --- Helper: calculate cost for K trips ---
        # We use a local cache to avoid recalculating the same k twice (optional but useful)
        def get_total_strategy_cost(k):
            gold_per_visit = total_gold / k
            
            # Create a SINGLE representative trip for this city with the given gold_per_visit
            temp_trip = Trip(
                cities=[(city, gold_per_visit)], 
                problem=problem,
                path_matrix=path_matrix,
                use_precompute=use_precompute
            )
            # Total cost = cost of one trip * k
            return temp_trip.total_cost * k, temp_trip

        # --- BINARY SEARCH FOR MINIMUM ---
        low = 1
        high = max_search
        
        # Save the best trip found during the search to avoid recreating it at the end
        best_k = 1
        best_trip_obj = None
        min_cost = float('inf')

        while low < high:
            mid = (low + high) // 2
            
            # Calculate cost at 'mid' and 'mid+1' to understand the slope
            cost_mid, trip_mid = get_total_strategy_cost(mid)
            cost_next, _       = get_total_strategy_cost(mid + 1)
            
            # Update the absolute best if we encounter it
            if cost_mid < min_cost:
                min_cost = cost_mid
                best_k = mid
                best_trip_obj = trip_mid
            
            if cost_next < min_cost: # Defensive check on next
                 min_cost = cost_next
                 best_k = mid + 1
                 # We don't save the 'next' object for efficiency, we recreate it only if it wins at the end
                 # (or we could save it, but it complicates the loop)
            # Binary Search Logic for Convex Functions
            if cost_mid < cost_next:
                # The curve rises to the right, the minimum is to the left (or is mid)
                high = mid
            else:
                # The curve descends to the right, the minimum is further ahead
                low = mid + 1
        
        # At the end of the loop, 'low' is the optimum (or very close)
        final_k = low
        final_cost, final_trip = get_total_strategy_cost(final_k)
        
        # Final check: better the one found in the loop or the "best" tracked?
        # (Generally they coincide, but for safety we take the best)
        if final_cost < min_cost:
            winner_trip = final_trip
            winner_k = final_k
        else:
            # If we had found a better point before (rare but possible at the edges)
            # If best_trip_obj is None (e.g., loop never entered), use final
            if best_trip_obj is None:
                winner_trip = final_trip
                winner_k = final_k
            else:
                winner_trip = best_trip_obj
                winner_k = best_k

        # --- Final List Creation ---
        # We create a trip with the optimal gold per visit,
        # repetaed 'winner_k' times
        best_trip = winner_trip.change_times_taken(winner_k)  # This updates the total cost inside the trip object
        best_trips.append(best_trip)
        
    return Solution(best_trips)