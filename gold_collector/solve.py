from gold_collector.core import Solution, Trip
from gold_collector.utils import generate_baseline

from icecream import ic

from copy import copy

import logging
from itertools import combinations


def solve(problem: Problem) -> Solution:
    sol = generate_baseline(problem)

    tested = set()

    # Iteratively improve by combining trips without mutating 'trips' during iteration.
    improved = True
    while improved:
        improved = False
        # Use a snapshot for combinations to avoid stale references when we rebuild the list.
        for trip1, trip2 in combinations(list(sol.trips), 2):
            combined_trip = Solution.combine_trips(copy(trip1), copy(trip2), problem)
            if combined_trip.total_cost < trip1.total_cost + trip2.total_cost:
                # Rebuild trips excluding the two combined ones by identity, then append the new combined trip.
                sol.trips = [t for t in sol.trips if t is not trip1 and t is not trip2]
                sol.trips.append(combined_trip)
                sol = Solution(trips=sol.trips)
                print(f"Combined trips {trip1.cities} and {trip2.cities} into {combined_trip.cities} with cost {combined_trip.total_cost}")
                improved = True
                break  # Restart combinations with updated list

    return sol