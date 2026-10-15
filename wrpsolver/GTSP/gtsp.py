import numpy as np

from .glns.gtsp_glns import GtspGLNS
from .Astar.astar_cpp import record_distance_cpp

def get_trace_glns(
    tsp_case: tuple[list[tuple[int, int]], list[int], list[list[int]]],
    grid: np.ndarray,
    step: int,
) -> tuple[list[tuple[int, int]], float, list[tuple[int, int]]]:
    """
    Solve a GTSP instance and return the selected route with its path.

    Args:
        tsp_case: (city_position, goods_class, city_class) tuple, where city_position 
            is the list of city grid coordinates, goods_class assigns each city to a
            "goods" cluster, and city_class is currently unused here (kept for 
            signature compatibility with the caller).
        grid: 2D grid used for pathfinding.
        step: Step size passed through to the A* search.

    Returns:
        selected_points: grid coordinates of the cities visited, in tour
            order.
        cost: total tour cost returned by GLNS.
        path: single flattened list of grid cells forming the full route,
            with consecutive legs' shared endpoints de-duplicated.
    """
    city_position, goods_class, city_class = tsp_case
    city_num = len(city_position)
    goods_num = len(set(goods_class))

    path_list, distance = record_distance_cpp(
        city_position, grid, step, diagonal_movement=True
    )

    # GLNS solves a closed-loop GTSP (it always connects the tour's last
    # point back to the first, since that's the standard TSP formulation).
    # For this exploration use case we don't want that closing leg, but we
    # can't just discard it after the fact: if the return-to-start cost were
    # left low, the solver would treat it as "free" and could choose to
    # route through a longer, suboptimal path elsewhere in order to exploit
    # that low-cost shortcut back to city 0. Setting an artificially high
    # cost for "any city -> city 0" forces the solver to treat closing the
    # loop as expensive, so it optimizes the rest of the tour as if it were
    # open-ended. We then drop the (now-irrelevant) closing leg ourselves
    # further down, after the solver has already produced a correctly
    # optimized route.
    distance[1:, 0] = 100000
    cost, solution = GtspGLNS(city_num, goods_num, goods_class, city_class, distance)
    solution = [pos - 1 for pos in solution]
    # Build the path for each consecutive leg of the tour (wrapping around
    # at the end via `goods_num`), then drop the final wrap-around leg —
    # the artificially-expensive closing leg back to city 0 isn't a real
    # part of the route (see note above), so it's excluded from the output.
    path_list = [
        path_list[solution[i % goods_num]][solution[(i + 1) % goods_num]]
        for i in range(goods_num)
    ]
    path_list.pop()

    selected_points = [city_position[i] for i in solution]
    path = []
    for leg in path_list:
        path += leg[:-1]
    path.append(path_list[-1][-1])

    return selected_points, cost, path