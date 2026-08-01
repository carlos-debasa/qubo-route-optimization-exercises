# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## ------- import packages -------
import networkx as nx
import dimod
import itertools
from dimod import SimulatedAnnealingSampler


def get_qubo(G, lagrange, n):
    """Returns a dictionary representing a QUBO for the Traveling Salesperson Problem.

    Variables are indexed as (city, time) where city in 0..n-1 and time in 0..n-1.
    The QUBO includes:
      - path length objective: sum_t sum_{u!=v} w_uv * x_{u,t} x_{v,t+1}
      - penalty that each city appears exactly once: lagrange*(sum_t x_{v,t} - 1)^2
      - penalty that each time slot has exactly one city: lagrange*(sum_v x_{v,t} - 1)^2

    Returns Q (dict) and offset (float).
    """
    Q = {}

    def add_qubo(i, j, value):
        key = (i, j)
        # keep keys ordered for consistency
        if key in Q:
            Q[key] += value
        else:
            Q[key] = value

    # Objective: travel cost
    # For each ordered pair u != v, and each time t, add weight * x_{u,t} x_{v,t+1}
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        # add both orientations since the route is ordered (u at t then v at t+1, and v at t then u at t+1)
        for t in range(n):
            tnext = (t + 1) % n
            add_qubo((u, t), (v, tnext), w)
            add_qubo((v, t), (u, tnext), w)

    # Penalty terms
    # Each city appears exactly once (across times): (sum_t x_{v,t} - 1)^2
    # Each time has exactly one city (across cities): (sum_v x_{v,t} - 1)^2
    # Expanding these produces linear and quadratic terms. The constant offsets are collected separately.

    # Linear contribution from constraints and quadratic between variables in the same constraint
    # After expansion, for each variable the linear contribution from one constraint is -lagrange,
    # and each pair in the same constraint gets +2*lagrange.

    # Add linear -2*lagrange for each variable (since there are two constraints per variable)
    for city in range(n):
        for t in range(n):
            add_qubo((city, t), (city, t), -2 * lagrange)

    # Off-diagonal penalty: pairs of different times for the same city
    for city in range(n):
        for t1, t2 in itertools.combinations(range(n), 2):
            add_qubo((city, t1), (city, t2), 2 * lagrange)

    # Off-diagonal penalty: pairs of different cities for the same time
    for t in range(n):
        for c1, c2 in itertools.combinations(range(n), 2):
            add_qubo((c1, t), (c2, t), 2 * lagrange)

    # The constant offset from expanding both sets of constraints is 2 * n * lagrange
    offset = 2 * n * lagrange

    return Q, offset


def get_sampler():
    """Returns a classical sampler (Simulated Annealing)."""
    sampler = SimulatedAnnealingSampler()
    return sampler


## ------- Main program -------
if __name__ == "__main__":

    lagrange = 4000
    n = 7
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 2230),
        (0, 2, 1631),
        (0, 3, 1566),
        (0, 4, 1346),
        (0, 5, 1352),
        (0, 6, 1204),
        (1, 2, 845),
        (1, 3, 707),
        (1, 4, 1001),
        (1, 5, 947),
        (1, 6, 1484),
        (2, 3, 627),
        (2, 4, 773),
        (2, 5, 424),
        (2, 6, 644),
	(3, 4, 302),
	(3, 5, 341),
	(3, 6, 1027),
	(4, 5, 368),
	(4, 6, 916),
	(5, 6, 702)
    ])
    Q, offset = get_qubo(G, lagrange, n)
    sampler = get_sampler()
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
    response = sampler.sample(bqm, label="Training - TSP")

    start = None
    sample = response.first.sample
    cost = response.first.energy
    route = [None] * n

    for (city, time), val in sample.items():
        if val:
            route[time] = city

    if start is not None and route[0] != start:
        # rotate to put the start in front
        idx = route.index(start)
        route = route[idx:] + route[:idx]

    if None not in route:
        print(route)
        print(cost)
