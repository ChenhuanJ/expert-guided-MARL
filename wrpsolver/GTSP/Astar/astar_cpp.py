from typing import Any

import numpy as np
import numpy.typing as npt

from . import Astar_cpp

# A single path is a list of [row, col] grid coordinates.
Path = list[list[int]]

def record_distance_cpp(
    city_position: list[list[int]],
    grid: npt.NDArray[np.int_],
    step: int,
    diagonal_movement: bool,
) -> tuple[list[list[Path]], npt.NDArray[np.int_]]:
    """
    Compute all-pairs shortest paths and distances between cities.

    Calls into the C++ extension (`Astar_cpp.GetPath`) to get a raw
    path/distance matrix, then symmetrizes it — see note below.

    Args:
        city_position: (city_num, 2) array of city (row, col) grid coords.
        grid: 2D grid used for pathfinding (obstacles / walkable cells).
        step: Step size passed through to the A* search.
        diagonal_movement: Whether diagonal moves are allowed.

    Returns:
        path: city_num x city_num nested list; path[i][j] is the list of
            grid cells from city i to city j.
        distance: (city_num, city_num) int ndarray of path costs,
            guaranteed symmetric (distance[i, j] == distance[j, i]).
    """
    path, distance = Astar_cpp.GetPath(grid, city_position, step, diagonal_movement)

    # NOTE: the C++ side computes each direction (i -> j and j -> i)
    # independently across worker threads with no synchronization, so on
    # grids with heuristic ties it can occasionally return two valid but
    # different-length shortest paths for the same pair, i.e. distance[i, j]
    # != distance[j, i]. This loop reconciles that: for each pair, keep
    # whichever direction found the shorter path and mirror it onto the
    # other entry, so the returned path/distance matrices are consistent.
    distance = np.array(distance).astype(int)
    for i in range(len(path)):
        for j in range(i + 1, len(path)):
            if distance[i, j] < distance[j, i]:
                path[j][i] = path[i][j][::-1]
            else:
                path[i][j] = path[j][i][::-1]
    distance = np.minimum(distance, distance.T)

    return path, distance


def compute_distance(
    start_position: list[int],
    end_position: list[int],
    grid: npt.NDArray[np.int_],
    step: int = 1,
    is_diagonal: bool = True,
) -> float:
    """
    Compute the shortest-path cost between two points on a grid.

    Args:
        start_position: (row, col) starting grid coordinate.
        end_position: (row, col) target grid coordinate.
        grid: 2D grid used for pathfinding.
        step: Step size passed through to the A* search.
        is_diagonal: Whether diagonal moves are allowed.

    Returns:
        The path cost (float) as computed by A*.
    """
    cost = Astar_cpp.findPathBetween2Points(
        grid, start_position, end_position, is_diagonal, step
    )
    return cost