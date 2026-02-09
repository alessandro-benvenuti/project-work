# project-work

# Gold Collection Optimization: A Parallel Evolutionary Approach

**A high-performance solver for a constraint-based Vehicle Routing Problem with dynamic weight penalties.**

## 1. Project Overview

This project implements a **Parallel Genetic Algorithm (Island Model)** to solve a complex variation of the Traveling Salesperson Problem (TSP). In this scenario, an agent must collect gold from distributed cities in a connected graph and return it to a home base.

Unlike standard routing problems where edge weights are static (distance only), this problem introduces a **dynamic cost function**: the cost of traversing an edge increases non-linearly based on the accumulated gold (weight) carried by the agent.

### Key Features

* **Parallel Architecture**: Utilizes Python's `ProcessPoolExecutor` to run independent evolutionary "Islands" on separate CPU cores.
* **Ring Migration**: Implements a migration topology where islands periodically exchange their best solutions to prevent premature convergence.
* **Hybrid Initialization**: Seeds the population with solutions from multiple constructive heuristics (Adaptive Split, Topology Savings) to jump-start the evolutionary process.
* **Optimized Graph Access**: Uses monkey-patching and precomputed geometry to bypass standard library overhead for critical pathfinding operations.

---

## 2. Problem Formulation

The problem is modeled as a graph exploration task where the goal is to empty the gold reserves of all cities with the minimum possible energy cost.

### 2.1 The Graph Environment

* **Nodes**: The world is a connected graph $G(V, E)$.
  * Node $0$: The **Home Base**. All trips must start and end here.
  * Nodes $1 \dots N$: Cities, each containing a specific amount of gold $g_i > 0$.
* **Edges**: Connections between cities with a Euclidean distance $d_{ij}$.

### 2.2 The Objective

The agent must generate a **Solution** consisting of a sequence of trips. A valid solution must ensure that:

1. Every trip starts and ends at Node 0.
2. The sum of gold collected from a city across all trips equals the city's initial gold .
3. The **Total Cost** is minimized.

### 2.3 The Physics (Cost Function)

The core challenge lies in the cost function. The cost to travel between two nodes is not just the distance; it is a function of distance and the current **payload** (gold on the truck).

For a path segment of length $d$ carrying weight , the cost is calculated as:

$$\text{Cost}(d, w) = d + ( \alpha \cdot d \cdot w )^\beta$$

Where:

* $d$: Euclidean distance of the edge.
* $w$: Current accumulated gold on the truck.
* $\alpha$: A scalar constant (Weight Penalty Factor).
* $\beta$: An exponent controlling the penalty severity.

### 2.4 The Optimization Challenge

This formula creates a dynamic trade-off that changes based on the parameters:

* **When $\beta > 1$ (High Penalty)**: The cost grows exponentially with weight.
* *Implication*: Long paths with heavy loads are prohibitively expensive. The optimal strategy often involves making many short trips (returning to base frequently) or "splitting" a city's gold into multiple visits to keep  low.


* **When $\beta \le 1$ (Low Penalty)**: The cost is linear or sub-linear.
* *Implication*: Distance dominates the cost. The problem behaves more like a traditional TSP/VRP, where visiting as many cities as possible in a single loop is optimal to minimize total distance traveled.

---

## 3. Solution Architecture

The solution adopts a **Hybrid Meta-heuristic** approach. It combines constructive heuristics (to generate high-quality initial seeds) with a Parallel Genetic Algorithm (to refine solutions through evolution).

### 3.1 Strategy Selection

The solver analyzes the problem parameters (specifically ) to choose the optimal strategy:

* **High Beta ($\beta \ge 5$)**: The cost is dominated by the weight penalty. The solver bypasses the GA and uses a specialized "Adaptive Split" heuristic to minimize weight carried.
* **Low/Medium Beta ($\beta < 5$)**: The solver launches the **Archipelago**, a parallel evolutionary system.

### 3.2 Constructive Heuristics (Initialization)

Before evolution begins, we generate diverse "seed" solutions using three distinct algorithms:

1. **Baseline**: One trip per city (naive).
2. **Topology Savings**: Merges trips based on graph proximity.
3. **Adaptive Split**: A custom heuristic that uses **binary search** to find the optimal number of visits () for each city. It balances the overhead of extra distance against the savings from reduced weight penalties.

**Code Snippet: Adaptive Split Logic (Simplified)**

```python
def generate_adaptive_split(problem):
    """
    Determines the optimal number of trips (k) for each city using Binary Search.
    """
    trips = []
    for city in range(1, problem.num_cities):
        # Binary Search for the optimal 'k' splits
        low, high = 1, 50
        best_k = 1
        min_cost = float('inf')

        while low < high:
            mid = (low + high) // 2
            # Compare cost of 'mid' vs 'mid + 1' splits
            cost_mid = calculate_split_cost(city, mid, problem)
            cost_next = calculate_split_cost(city, mid + 1, problem)

            if cost_mid < cost_next:
                high = mid
                best_k = mid
            else:
                low = mid + 1
                best_k = mid + 1
        
        # Create the optimal trip configuration
        trips.append(create_trip(city, splits=best_k))
    return Solution(trips)

```

### 3.3 The Island Model (Parallel GA)

To utilize multi-core processors and maintain population diversity, the solution implements an **Island Model**.

* **Islands**: Independent populations evolving on separate CPU cores. Each island is initialized with a different seed strategy:
  * Island 0 uses the Baseline for initialization and a random mutation strategy,
  * Island 1 uses the Topology Savings for initialization and a merging mutation strategy,
  * Island 2 uses Adaptive Split for initialization and an "adjustive repeats" mutation strategy,
  * Island 3 uses Adaptive Split for initialization and a random mutation strategy.
* **Ring Migration**: Every 20 generations, islands pause to exchange genetic material. The best individual from Island $i$ replaces the worst individual in Island $i+1$.

**Code Snippet: Ring Migration**

```python
def perform_migration(self):
    """
    Executes Ring Migration: Island 0 -> Island 1 -> ... -> Island 0
    """
    # 1. Harvest the Elites (Deepcopy to prevent reference issues)
    migrants = [deepcopy(island.population[0]) for island in self.islands]

    # 2. Inject into Neighbors
    for i in range(len(self.islands)):
        target_idx = (i + 1) % len(self.islands)
        target_island = self.islands[target_idx]
        migrant = migrants[i]

        # Replace the worst individual with the elite migrant
        target_island.population[-1] = migrant
        
        # Re-sort population by cost
        target_island.population.sort(key=lambda x: x.cost)

```

### 3.4 Genetic Operators

The evolutionary process refines the solution using custom mutation and crossover operators designed for this specific routing problem.

#### The Genotype

A `Solution` is represented as a list of `Trip` objects. Each `Trip` is a sequence of visited cities and the amount of gold collected.

#### Mutation Strategies

The mutation operator randomly selects one of five strategies to alter a solution:

1. **Merge**: Combines two trips into one (reducing distance).
2. **Split**: Breaks a trip into two (reducing weight penalty).
3. **Swap**: Exchanges cities between two trips.
4. **Gold Split**: Redistributes the gold collection of a single city across two trips.
5. **Adjust Repeats**: Optimizes how many times a specific route is repeated.

**Code Snippet: Mutation Router**

```python
def mutate(self, problem, strategy="random"):
    if strategy == "random":
        strategy = random.choice(["merge", "split", "swap", "gold_split", "adjust_repeats"])
    
    if strategy == "merge":
        # Combine two random trips
        t1, t2 = random.sample(self.trips, 2)
        new_trip = combine_trips(t1, t2)
        # Update solution...
        
    elif strategy == "split":
        # Break a heavy trip into two lighter ones
        trip = random.choice(self.trips)
        t1, t2 = split_trip(trip)
        # Update solution...
    
    # ... handle other strategies
    return self

```

#### Crossover Strategy (Greedy Inheritance)

The crossover operator is designed to preserve high-quality sub-tours while repairing valid constraints. It combines two parent solutions to produce a robust offspring.

1. **Inheritance from Parent A**: The child inherits 50% of trips directly from the first parent, locking in their specific gold collection amounts.
2. **Greedy Fill from Parent B**: The remaining trips are taken from the second parent, but only if they collect gold from cities that are not yet "full." If a trip from Parent B collects too much gold for a specific city, it is dynamically resized (number of repeats reduced) to fit the remaining capacity.
3. **Sweeper Repair**: Finally, if any small amounts of gold remain uncollected (dust), the operator creates dedicated "Sweeper Trips" to ensure the solution is valid and 100% complete.

**Code Snippet: Crossover Logic**

```python
def crossover(self, other_parent, problem):
    # 1. Inherit half of trips from Parent A
    child_trips = random.sample(self.trips, len(self.trips) // 2)
    collected_gold = track_gold(child_trips)

    # 2. Greedily take useful trips from Parent B
    for trip in other_parent.trips:
        if is_useful(trip, collected_gold):
            # Adjust 'times_taken' to fit remaining gold capacity
            adjusted_trip = fit_to_capacity(trip, collected_gold)
            child_trips.append(adjusted_trip)

    # 3. Add "Sweeper Trips" for any missing dust
    missing_gold = calculate_missing(child_trips)
    for city, amount in missing_gold:
         child_trips.append(create_sweeper_trip(city, amount))
    
    return Individual(child_trips)

```

### 3.5 Algorithmic Decisions & Tuning
The evolutionary algorithm utilizes a fixed parameter schedule (200 generations, offspring size 20, tournament size 3, and mutation probability 0.2) rather than adaptive or decaying rates.<br>
This design choice stems from our initialization strategy. Since the population is seeded with strong heuristics (Adaptive Split, Topology Savings), the initial solutions are already positioned in deep local optima. We observed that:
- Exploitation Risks: A traditional "exploitation-heavy" approach (e.g., low mutation, high selection pressure) caused immediate stagnation, as the algorithm was unable to escape the basins of attraction formed by the heuristic seeds.
- Exploration Necessity: By maintaining high exploration pressure throughout the entire run (via diverse mutation operators and fixed tournament sizes), we force the algorithm to break structural constraints and discover novel improvements that the heuristics missed.

While further hyperparameter tuning could potentially yield marginal gains, this configuration provided the most robust balance between convergence speed and solution quality during our experiments.

---

## 4. Technical Optimizations

To handle large graphs ($N=1000$) within the strict time constraints, several low-level optimizations were implemented to bypass standard library overheads.

### 4.1 Monkey Patching for Speed

The provided `Problem` class originally returned a *copy* of the graph every time `problem.graph` was accessed. In an evolutionary loop accessing the graph millions of times, this copying becomes a massive bottleneck.

We applied a **Monkey Patch** to overwrite the property accessor, forcing it to return the private reference `_graph` directly. This simple change yielded a significant speedup (orders of magnitude) without altering the core logic.

**Code Snippet: The Patch**

```python
# --- MONKEY PATCH ---
# Overwrite the 'graph' property to avoid expensive copying
def fast_graph_accessor(self):
    return self._graph  # Return reference, not copy

Problem.graph = property(fast_graph_accessor)

```

### 4.2 Precomputed Weighted Paths (Fixed Geometry)

There are scenarios where recalculating the optimal path for every fitness evaluation is either unnecessary or too slow:
1. For $\beta \le 1$ there is no exponential increas of the cost, so it would be pointless.
2. For $N > 200$ recalculating the optimal path would be too slow, since the complexity is $O(E \log V)$.

We implemented a **"Fixed Geometry"** heuristic. Instead of running A* dynamically, we run All-Pairs Dijkstra **once** at the start. The edge weights are calculated assuming the truck carries a "reference load" (50% of the average city gold). This pre-calculates a lookup table of optimal paths, reducing route generation from $O(E \log V)$ to $O(1)$.

**Code Snippet: Precomputation Logic**

```python
def precompute_weighted_paths(problem, ref_gold_ratio=0.5):
    """
    Bakes the dynamic cost formula into static edge weights 
    using a heuristic "average load" to pre-calculate paths.
    """
    avg_gold = total_gold / problem.num_cities
    ref_load = avg_gold * ref_gold_ratio
    
    # Create static graph with weighted edges
    G_static = problem.graph.copy()
    for u, v, data in G_static.edges(data=True):
        d = data['dist']
        # Bake cost formula into weight
        data['weight'] = d + (problem.alpha * d * ref_load) ** problem.beta

    # Run All-Pairs Dijkstra once
    return dict(nx.all_pairs_dijkstra_path(G_static, weight='weight'))

```

### 4.3 Parallel Execution

The solution leverages Python's `ProcessPoolExecutor` to run the **Island Model** in parallel. Each "Island" evolves on a separate CPU core, and migration is handled by the main process during synchronization points. This maximizes hardware utilization and allows for a larger total population size.


---

## 5. Project Structure

The project is organized into a modular package structure to separate core logic, heuristics, and the evolutionary engine.

```plaintext
.
├── logs/                   # Directory for runtime logs and results CSVs
├── src/                    # Main source code (formerly gold_collector)
│   ├── core.py             # Core data structures: Solution, Trip, and Path calculations
│   ├── genetic.py          # Evolutionary logic: Individual, Population, Mutation, Crossover
│   ├── solve.py            # Main solver orchestration: Archipelago, Migration, Parallel Execution
│   └── utils.py            # Constructive Heuristics: Adaptive Split, Topology Savings, Precomputation
├── s343748.py              # Entry point script (Main Execution & Monkey Patching)
└── README.md               # Project documentation

```

### Key Modules

* **`src/core.py`**: Defines the `Trip` and `Solution` classes. It handles the low-level cost calculations and exact path verification using Dijkstra/A*.
* **`src/genetic.py`**: Contains the `Island` class and the genetic operators (`mutate`, `crossover`). This file defines how solutions evolve over generations.
* **`src/utils.py`**: A collection of high-performance heuristics used to seed the population and optimize geometry. Includes the `generate_adaptive_split` and `precompute_weighted_paths` functions.
* **`src/solve.py`**: The "Brain" of the operation. It decides which strategy to run (Heuristic vs. Genetic) based on problem parameters ($\beta$) and manages the parallel `ProcessPoolExecutor`.

---

## 7. Experimental Results

We evaluated the solver on **256 different problem configurations**, varying the number of cities ($N$), graph density, and cost function parameters ($\alpha, \beta$).

The full raw data can be found here: [📄 logs/results.csv](logs/results.csv)

### 6.1 Summary Metrics

#### Impact of Problem Size ($N$)
As the number of cities increases, the problem complexity grows exponentially. The Parallel Genetic Algorithm maintains solution quality even for large instances ($N = 1000$), though computation time increases naturally (while always being on average under 10 minutes).

| num_cities | Avg Improvement (%) | Avg Time (s) |
| --- | --- | --- |
| 50 | 57.63 | 27.60 |
| 100 | 57.51 | 25.33 |
| 200 | 56.06 | 46.55 |
| 1000 | 54.53 | 573.54 |

#### Impact of Graph Density
This parameters determines how well the graph is connected.
| density | Avg Improvement (%) | Avg Time (s) |
| --- | --- | --- |
| 0.10 | 58.61 | 87.02 |
| 0.20 | 58.01 | 96.71 |
| 0.50 | 56.63 | 173.45 |
| 1.00 | 52.47 | 315.83 |

#### Impact of Cost Multiplier ($\alpha$)
This parameter corresponds to the cost penality factor.
| alpha | Avg Improvement (%) | Avg Time (s) |
| --- | --- | --- |
| 0.10 | 55.65 | 155.81 |
| 1.00 | 56.71 | 154.47 |
| 2.00 | 56.68 | 157.70 |
| 5.00 | 56.68 | 205.04 |

#### Impact of Cost Exponent ($\beta$)
This is the most critical parameter. It dictates whether the problem deals with linear or non-linear costs. Indeed as seen before most of the logic behind this project is focused on finding strategies to handle different values of $\beta$.
| beta | Avg Improvement (%) | Avg Time (s) |
| --- | --- | --- |
| 0.10 | 27.02 | 139.42 |
| 1.00 | 0.22 | 141.88 |
| 2.00 | 98.47 | 292.79 |
| 5.00 | 100.00 | 98.93 |


### 6.2 Analysis of Results

- **1. Consistent Performance at Scale**
    The data demonstrates that the solver is highly scalable. Independent of the problem size (), graph density, or the linear weight factor , the algorithm consistently finds solutions that are **54% to 58% better** than the baseline. This proves that the core logic—optimizing path geometry via the Island Model—is effective across a wide range of graph structures.

- **2. The Beta Regimes**
    The value of  fundamentally changes the optimal strategy, and our results reflect three distinct behaviors:

    1. <strong> $\beta<1$ Distance Dominates</strong>: The cost function is sub-linear regarding weight. The problem behaves similarly to a standard Traveling Salesperson Problem (TSP). The baseline strategy (returning to base after every city) maximizes travel distance and is therefore extremely inefficient. Our Genetic Algorithm exploits this by linking many cities into single tours, achieving significant savings (~27%).
    2.  <strong> $\beta \approx 1$ The Tipping Point</strong>: At , the cost of carrying weight scales linearly with distance. In this specific regime, the baseline strategy of "emptying the truck" after every city is actually a very strong local optimum, as it minimizes the weight-distance product. Our solver matches this performance (0.22% improvement), indicating that the baseline is near-optimal for this specific physics configuration.
    3.  <strong> $\beta>1$ Weight Dominates</strong>: The penalty for carrying weight becomes exponential. The baseline strategy fails catastrophically here because it forces the agent to carry a *full* city's gold load at once. Our **Adaptive Split** heuristic dominates in this regime by identifying that it is cheaper to visit a city multiple times (carrying small fractions of gold) than to visit it once. This strategy yields massive cost reductions, approaching **99-100% improvement** over the naive approach.


---

## 7. Installation & Usage

### Prerequisites

* Python 3.10+
* Dependencies: `numpy`, `matplotlib`, `networkx`, `icecream`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/gold-collection-optimization.git
cd gold-collection-optimization

```


2. Install the required packages:
```bash
pip install numpy matplotlib networkx icecream

```



### Running the Solver

To execute the main script and run the experiments across all parameter combinations:

```bash
python s343748.py

```

The script will:

1. Generate problems with varying cities ($50, 100, 200, 1000$) and parameters ($\alpha, \beta$).
2. Run the **Parallel Island Model** or **Adaptive Split Heuristic** depending on complexity.
3. Log detailed results (Baseline Cost vs. Solution Cost) to `logs/results.csv`.

---


## 8. Acknowledgments

This project was developed individually, with the following notes on collaboration and tools:

* **Peer Collaboration**: I engaged in high-level conceptual discussions with three colleagues. We shared ideas regarding algorithmic strategies (e.g., the viability of Genetic Algorithms vs. Simulated Annealing) and heuristic approaches. **No code was shared or copied**; all implementations, architecture, and fine-tuning are unique to this repository. 
  * Irene Bartolini 345905
  * Michele Carena s349483
  * Davide Carletto s339425
  
* **AI Assistance**: Large Language Models (LLMs) were utilized as supportive tools during development. They were primarily used for:
  * **Debugging**: Identifying subtle logic errors in complex pathfinding functions.
  * **Refactoring**: Optimizing specific code blocks for better performance.
  * **Boilerplate**: Generating standard plotting and logging utility code.



---








### Repository Setup

1. Create a Git repository named project-work.
2. Inside the repository, include:
    - A Python file named s<student_id>.py.
    - A folder named src/ containing all additional code required to run your solution.
    - A TXT file named base_requirements.txt containing the basic python libraries that you need to run the code to generate the problem.


### Main File Requirements (s<student_id>.py)

1. Define a class responsible for generating the problem and storing the best solution found.
2. Implement a method called solution() that returns the optimal path in the following format: 
```python
[(c1, g1), (c2, g2), …, (cN, gN), (0, 0)]
```
where:
- c1, …, cN represent the sequence of cities visited.
- g1, …, gN represent the corresponding gold collected at each city.

qui
### Rules
1. The thief must start and finish at (0, 0).
2. Returning to (0, 0) during the route is allowed to unload collected gold before continuing.
3. Don't forget to change the name of the file s123456.py provided as an example ;).

### Notes
- It is not necessary to push the report.pdf or log.pdf in this repo.
- It is mandatory to upload it in "materiale" section of "portale della didattica" at least 168 hours before the exam call.
- For well commented codes, I can't ensure a higher mark but they would be very welcome.
- In case you face any issue or you have any doubt text me at the email giuseppe.esposito@polito.it and professor Squillero giovanni.squillero@polito.it.