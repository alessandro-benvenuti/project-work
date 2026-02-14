import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

import os
import csv
import time
import itertools

from src.core import Trip, Solution
from src.solve import solve

from Problem import Problem

# --- MONKEY PATCH START ---
# We overwrite the 'graph' property of the Problem class.
# Instead of creating a copy (nx.Graph(self._graph)), we return the private reference directly.
def fast_graph_accessor(self):
    return self._graph          # This way we do not create a copy each time, making it extremely faster.

def get_num_cities(self):
    return self._graph.number_of_nodes()

Problem.graph = property(fast_graph_accessor)
Problem.num_cities = property(get_num_cities)
# --- MONKEY PATCH END ---
    
def solution(problem: Problem) -> Solution:
    return solve(problem)[0]
    

def is_valid(path, p:Problem):
    for (c1, _), (c2, _) in zip(path, path[1:]):
        if not nx.has_path(p.graph, c1, c2):
             return False # Esce solo se trova un errore
    return True # Restituisce True solo alla fine del ciclo
