import math
import random

import numpy as np
import shapely

def get_line_list(
    lines: "shapely.LineString | shapely.LinearRing | shapely.MultiLineString",
) -> list[shapely.LineString]:
    """
    Normalize a shapely line-like geometry into a flat list of lines.

    """
    line_list = []
    if isinstance(lines, (shapely.LineString, shapely.LinearRing)):
        line_list.append(lines)
    elif isinstance(lines, shapely.MultiLineString):
        for line in lines.geoms:
            line_list.append(line)
    return line_list


def get_sample(
    convex_set: list[shapely.Polygon],
    init_polygon: shapely.Polygon,
    d_sample: float,
    grid_map: np.ndarray,
    step: int = 3,
) -> list[list[tuple[int, int]]]:
    """
    Sample candidate city points along the boundary of each convex region.

    For each polygon in convex_set, points are sampled at fixed intervals
    along the portion of its boundary that isn't part of the overall map
    boundary, then snapped onto the nearest valid step-aligned grid cell
    so they're compatible with the grid used by A* pathfinding.

    """
    sample_list = []

    # Shrink the map polygon inward slightly to define "safely walkable"
    # free space, and buffer its boundary outward to define an "obstacle"
    # zone near the outer edge that sample points should avoid.
    free_space = init_polygon.buffer(-1, join_style=2)
    obstacle = init_polygon.boundary.buffer(1, join_style=2)

    for polygon in convex_set:
        point_list = []
        temp_point_list = []

        # Only sample the parts of this polygon's boundary that aren't
        # near the overall map boundary.
        line_list = get_line_list(polygon.boundary.difference(obstacle))

        for line in line_list:
            path = d_sample
            while path < line.length:
                point = shapely.line_interpolate_point(line, path)
                if free_space.contains(point):
                    temp_point_list.append(point)
                path += d_sample

            # Always include the line's end point too, if it's valid.
            end_point = shapely.line_interpolate_point(line, 1)
            if free_space.contains(end_point):
                temp_point_list.append(end_point)

        for point in temp_point_list:
            # Snap the sampled (continuous) point onto the nearest
            # step-aligned grid cell. Since the point may not land exactly
            # on a grid line, consider all 4 surrounding grid-aligned
            # candidates and pick the first one that's walkable.
            x1 = math.ceil(point.x / step) * step
            x2 = x1 - step
            y1 = math.ceil(point.y / step) * step
            y2 = y1 - step
            candidates = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
            random.shuffle(candidates)

            for pos in candidates:
                in_bounds = (
                    0 < pos[0] < grid_map.shape[1]
                    and 0 < pos[1] < grid_map.shape[0]
                )
                if in_bounds and grid_map[pos[1]][pos[0]] != 0:
                    point_list.append(pos)
                    break

        # Remove duplicate points while preserving first-seen order.
        # (dict preserves insertion order in Python 3.7+; using it as an
        # ordered set here.)
        point_list = list(dict.fromkeys(point_list))

        if len(point_list) > 0:
            sample_list.append(point_list)

    return sample_list


def post_processing(
    sample_list: list[list[tuple[int, int]]],
) -> tuple[list[tuple[int, int]], list[int], list[list[int]]]:
    """
    Flatten per-polygon sample groups into GTSP-ready city data.

    """
    city_pos = []
    city_goods = []
    city_class = []
    n = 0

    for sample in sample_list:
        classify = []
        for point in sample:
            city_pos.append((point[0], point[1]))

            city_goods.append(sample_list.index(sample))

            classify.append(n)
            n += 1

        if len(classify) > 0:
            city_class.append(classify)

    return city_pos, city_goods, city_class
