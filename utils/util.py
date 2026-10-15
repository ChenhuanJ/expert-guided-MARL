import numpy as np
from shapely.geometry import Polygon
from scipy.ndimage import distance_transform_edt
from shapely.vectorized import contains

Position = list[int]

def get_start_points(
    polygon: Polygon,
    close_prob: float,
    num_agents: int,
    max_distance: int = 40,
    close_radius: int = 5,
) -> list[Position]:
    """
    Randomly select starting points for robots.

    Args:
        polygon:
            The environment polygon.
        close_prob:
            Probability of sampling all robots close to each other.
        num_agents:
            Number of robots to initialize.
        max_distance:
            Maximum distance from the polygon boundary for candidate points.
            Larger values allow starting points farther from the boundary.
        close_radius:
            Maximum distance between the center of a close spawn group and each
            starting point.

    """
    max_x = polygon.bounds[2]
    max_y = polygon.bounds[3]
    xs, ys = np.meshgrid(
        np.arange(max_x, dtype=np.float64), np.arange(max_y, dtype=np.float64)
    )
    mask = contains(polygon, xs, ys)
    distance_map = distance_transform_edt(mask)  # type: ignore

    y, x = np.where(
        (mask) & (distance_map > 0) & (distance_map < max_distance)
    )

    if len(x) < num_agents:
        raise RuntimeError(
            f"Only {len(x)} valid starting points found for {num_agents} agents."
        )

    
    if np.random.rand() < close_prob:
        candidate_points = np.column_stack((x, y))
        max_tries = 50

        for _ in range(max_tries):
            center_idx = np.random.randint(len(candidate_points))
            center = candidate_points[center_idx]

            offsets = candidate_points - center
            distances = np.linalg.norm(offsets, axis=1)
            nearby = np.flatnonzero(distances <= close_radius)

            if len(nearby) < num_agents:
                continue

            selected = np.random.choice(nearby, size=num_agents, replace=False)
            return candidate_points[selected].tolist()

        raise RuntimeError(
            f"Failed to find {num_agents} nearby starting points within "
            f"radius {close_radius}."
        )
    
    selected = np.random.choice(len(x), size=num_agents, replace=False)

    return np.column_stack((x[selected], y[selected])).tolist()
        # tries = 0
        # directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # # Warning: This method is only valid for 4 robots or less
        # while tries < max_tries:
        #     idx = np.random.randint(0, len(x))
        #     start_points: list[list[int]] = [[x[idx], y[idx]]]
        #     for direction in directions:
        #         temp_x: int = int(x[idx] + robot_size * direction[0])
        #         temp_y: int = int(y[idx] + robot_size * direction[1])
        #         if (
        #             temp_y > max_y - robot_size or
        #             temp_y < 0 or
        #             temp_x > max_x - robot_size or
        #             temp_x < 0 or
        #             not polygon.contains(Point(temp_x, temp_y))
        #         ):
        #             continue
        #         if (
        #             mask[temp_y, temp_x]
        #             and distance_map[temp_y, temp_x] > robot_size
        #         ):
        #             start_points.append([temp_x, temp_y])
        #         if len(start_points) == num_agents:
        #             return start_points
        #     tries += 1
    # else:
    #     # Shuffle candidate indices
    #     idxs = np.random.permutation(len(x))

    #     start_points: list[list[int]] = []
    #     for i in idxs:
    #         px, py = x[i], y[i]

    #         # Boundary check
    #         if (
    #             py < robot_size or
    #             py > max_y - robot_size or
    #             px < robot_size or
    #             px > max_x - robot_size
    #         ):
    #             continue
    #         if not polygon.contains(Point(px, py)):
    #             continue

    #         start_points.append([px, py])
    #         if len(start_points) == num_agents:
    #             return start_points

    # raise RuntimeError("Failed to find starting points after max tries")
