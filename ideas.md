# Gold Collector Optimizer

This project tackles a graph-based routing problem: starting from city 0, collect all gold from the other cities while minimizing a cost function that penalizes distance and carried weight.

- Core file: [problem-2.ipynb](/Users/alessandrobenvenuti/Desktop/Università/Computational_intelligence/project/Gold-Collector-Optimizer/problem-2.ipynb)
- Main class: [`Problem`](/Users/alessandrobenvenuti/Desktop/Università/Computational_intelligence/project/Gold-Collector-Optimizer/problem-2.ipynb)
- Requirements overview: [requirements.md](/Users/alessandrobenvenuti/Desktop/Università/Computational_intelligence/project/Gold-Collector-Optimizer/requirements.md)

## Problem Overview

- Cities are nodes in a connected graph with positions and gold amounts.
- All routes start and end at city 0.
- Travel cost depends on distance and carried weight:
  Cost: $d + (d \cdot \alpha \cdot w)^\beta$

## Baseline Strategy

A naive baseline visits each city individually:
- From city 0 to a target city, collect its gold, then return to city 0.
- Total cost is the sum of the forward and return path costs for all cities.

This baseline provides a reference to beat with smarter multi-city trip planning.

## Approach: Island-Model Evolutionary Optimization

We use a multi-population evolutionary algorithm (island model) to explore diverse solution spaces and avoid premature convergence.

- Representation
  - A solution is a sequence of trips: [trip1, trip2, ...], where a Trip is an object with parameters:
    - <strong>path</strong>: a path with the format ```[(0,0), (c1, g1), (c2, g2), …, (cN, gN), (0, 0)]``` always starting and ending in the node 0
    - <strong>cities</strong>: the cities in which the gold is takne gold in the same order in which they are visited
  - Each trip is an ordered list of cities (excluding 0), implicitly starting and ending at 0.
  - Fitness is computed by simulating the path costs.

- Initialization (Diverse Seeds per Island)
  - Each island starts from a different heuristic seed (e.g., nearest-neighbor, baseline, clustering/TSP-inspired).
  - This creates complementary “styles” of solutions (short trips, long tours, weight-aware grouping).

- Variation Operators
  - Crossover: swap whole trips between parents; optionally edge-preserving crossover within trips.
  - Local search: 2-opt within a trip to reduce distance, path relinking between similar trips.
  - Mutations:
    - Intra-trip: swap cities, insert/delete, 2-opt.
    - Inter-trip: move city between trips, split/merge trips to balance weight vs distance.
  - Repair heuristics: split trips that exceed weight/penalty thresholds; merge short trips to reduce overhead.

- Selection and Diversity
  - Tournament selection with small k to maintain diversity.
  - Optional niching/penalties on very similar solutions to spread the search.

- Migration
  - Low-frequency migration (e.g., every 25–50 generations).
  - Send top-1/2 individuals to neighboring islands (ring or fully connected).
  - Replace worst or inject into population with re-evaluation.

- Stopping and Evaluation
  - Run until convergence or a generation/time limit.
  - Compare best solutions to baseline across varying density, α, β.
  - Track metrics: total cost, number of trips, average trip weight, improvement over baseline.
  
## Heuristic ideas
- First heuristic: Savings + Light-First
  - Build initial solution:
    - Start with single-city trips: for each city, 0 → city → 0 via shortest paths (ensures feasibility on non-complete graphs).
    - Greedily merge trips using a weight-aware Clarke–Wright savings:
      - Base savings = d(0, end(T1)) + d(0, start(T2)) − d(end(T1), start(T2)) computed on shortest-path distances.
      - Subtract a penalty estimating the heavier final return when trips are merged (depends on α, β and combined gold).
      - Visit light cities first (ascending gold), and choose the last city as the one nearest to 0 to minimize the heavy return leg. Merge only if a connecting shortest path exists and the net score improves.

## Roadmap

1. Implement evaluators and solution representation for multi-trip paths.
2. Add seed generators for different heuristics (to be discussed and tuned).
3. Build island engine (populations, operators, migration).
4. Benchmark against baseline on multiple graph instances and parameter settings.
5. Visualize trips and paths for qualitative inspection.

## Getting Started

- Open `problem-2.ipynb` in VS Code and run cells.
- Ensure dependencies: NumPy, Matplotlib, NetworkX, IceCream. Optional: scikit-learn (for clustering).
- Use the plotting utilities in `Problem.plot()` to inspect graphs.

## Status

The island-model design is defined; next step is selecting and implementing the best heuristic seeds per island.