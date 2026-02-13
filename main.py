def is_valid(path, p:Problem):
    for (c1, _), (c2, _) in zip(path, path[1:]):
        if not nx.has_path(p.graph, c1, c2):
             return False # Esce solo se trova un errore
    return True # Restituisce True solo alla fine del ciclo


def main():

    problem = Problem(10, density=0.2, alpha=1, beta=1)
    sol = solution(problem)

    print(is_valid(sol, problem))

    print(sol)


    # num_cities_list = [50, 100, 200, 1000]
    # num_cities_list = [1000]
    # densities = [0.1, 0.2, 0.5, 1.0]
    # alphas = [0.1, 1, 2, 5]
    # betas = [0.1, 1, 2, 5]

    # # Output file setup
    # os.makedirs("logs", exist_ok=True)
    # csv_file = "logs/results.csv"
    
    # # Define columns
    # columns = [
    #     "num_cities", "density", "alpha", "beta", 
    #     "baseline_cost", "solution_cost", "improvement_percent", "time_taken_sec",
    #     "errors"
    # ]

    # # Create file and write header if it doesn't exist
    # if not os.path.exists(csv_file):
    #     with open(csv_file, mode='w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(columns)

    # print(f"--- Starting Experiments ---")
    # print(f"Results will be appended to: {csv_file}")
    
    # # Generate all combinations
    # combinations = list(itertools.product(num_cities_list, densities, alphas, betas))
    # total_runs = len(combinations)
    
    # for i, (n, d, a, b) in enumerate(combinations):
    #     print(f"\n[{i+1}/{total_runs}] Running: Cities={n}, Dens={d}, Alpha={a}, Beta={b}")
        
    #     try:
    #         # Problem  Setup
    #         problem = Problem(num_cities=n, density=d, alpha=a, beta=b)
            
    #         # Teacher Baseline
    #         baseline_cost = problem.baseline()
            
    #         # My Solution
    #         start_time = time.time()
    #         solution_path, solution_cost, is_valid, message = solve(problem)
    #         elapsed = time.time() - start_time

    #         # Handle invalid solutions
    #         if not is_valid:
    #             print(f"   -> SOLUTION INVALID: {message}")
    #             # Log invalid solutions with a special marker
    #             with open(csv_file, mode='a', newline='') as f:
    #                 writer = csv.writer(f)
    #                 writer.writerow([n, d, a, b, f"{baseline_cost:.2f}", "INVALID", "0", f"{elapsed:.2f}", f"{message}"])
    #             continue
            
    #         # 4. Calculate Statistics
    #         if baseline_cost > 0:
    #             improvement = ((baseline_cost - solution_cost) / baseline_cost) * 100
    #         else:
    #             improvement = 0.0

    #         # 5. Log to CSV
    #         row = [n, d, a, b, f"{baseline_cost:.2f}", f"{solution_cost:.2f}", f"{improvement:.2f}", f"{elapsed:.2f}", "-"]
            
    #         with open(csv_file, mode='a', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(row)
                
    #         print(f"   -> Result: Base={baseline_cost:.0f} | Sol={solution_cost:.0f} | Imp={improvement:.2f}% | Time={elapsed:.1f}s")

    #     except Exception as e:
    #         print(f"   -> ERROR: {e}")
    #         #  Log errors to the CSV or a separate file
    #         with open("logs/errors.csv", mode='a', newline='') as f:
    #             writer = csv.writer(f)
    #             writer.writerow([n, d, a, b, "ERROR", "ERROR", "0", "0", f"{str(e)}"])

    # problem = Problem(100, density=0.2, alpha=1, beta=1)

    # baseline = problem.baseline()
    # print(f"Teacher baseline: {baseline}")
    

    # sol, cost = solution(problem)
    # print(f"Solution total cost: {cost}")

    # improvement = (baseline - cost) / baseline * 100
    # print(f"Improvement over baseline: {improvement:.2f}%")

    # times_better = baseline / cost
    # print(f"Times better than baseline: {times_better:.2e}")

    # with open("logs/solution.txt", "w") as f:
    #     f.write(str(sol))


    # print(Problem(100, density=0.2, alpha=1, beta=2).baseline())
    # print(Problem(100, density=0.2, alpha=2, beta=1).baseline())
    # print(Problem(100, density=1, alpha=1, beta=1).baseline())
    # print(Problem(100, density=1, alpha=2, beta=1).baseline())
    # print(Problem(100, density=1, alpha=1, beta=2).baseline())
    # print(Problem(1_000, density=0.2, alpha=1, beta=1).baseline())
    # print(Problem(1_000, density=0.2, alpha=2, beta=1).baseline())
    # print(Problem(1_000, density=0.2, alpha=1, beta=2).baseline())
    # print(Problem(1_000, density=1, alpha=1, beta=1).baseline())
    # print(Problem(1_000, density=1, alpha=2, beta=1).baseline())
    # print(Problem(1_000, density=1, alpha=1, beta=2).baseline())


if __name__ == '__main__':
    main()