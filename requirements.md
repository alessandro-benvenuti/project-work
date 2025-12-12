 Project Description: Gold Collection Optimization Problem

## Problem Statement

This project involves solving a *graph-based optimization problem* where the goal is to efficiently collect gold from all cities in a network while minimizing the total travel cost.

## Problem Setup

- ⁠*Graph Structure*: We have a connected graph with cities as nodes
- ⁠*Starting Point*: All journeys must start from city 0 (the home base)
- ⁠*Gold Distribution*: Each city (except city 0) contains a certain amount of gold to be collected
- ⁠*Objective*: Collect all gold from all cities while minimizing the total cost

## Cost Function

The cost of traveling between cities depends on the distance and the weight being carried:

*Cost Formula*: $\text{Cost} = d + (d \cdot \alpha \cdot w)^\beta$

Where:
- $d$ = distance between cities
- ⁠$w$ = weight of gold being carried
- ⁠$\alpha$ = weight penalty factor
- ⁠$\beta$ = exponential penalty factor

## Strategic Considerations

1) ⁠*Route Planning*: Design efficient paths to visit all cities
2) ⁠*Load Management*: Decide when to return to base (city 0) to drop off collected gold
3) ⁠*Multiple Trips*: The strategy can involve multiple round trips to optimize the weight penalty
4) ⁠*Trade-off*: Balance between longer routes (more cities per trip) vs. shorter routes (less weight penalty)

## Goal

*Beat the baseline solution* by implementing an intelligent algorithm that outperforms the naive approach of visiting each city individually and returning to base after each collection.

The baseline strategy visits each city separately (go to city → collect gold → return to base), which may not be optimal due to the weight penalty in the cost function.