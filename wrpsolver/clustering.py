import math
import random
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
from sklearn.cluster import SpectralClustering

# import pymetis

from .GTSP.Astar.astar_cpp import compute_distance, record_distance_cpp

def get_centroids(convex_set: list[Polygon], grid_map: np.ndarray) -> list[list[int]]:
    ''' Return the centroid of the polygon list preserving its order '''
    point_list = []
    for polygon in convex_set:
        centroid = polygon.centroid
        point = [centroid.x, centroid.y]
        grid_pos = convert_position_to_grid(point, grid_map)
        point_list.append(grid_pos)

    # NOTE: dedup by value — see docstring warning above about index
    # alignment with convex_set.
    point_list = list(dict.fromkeys(point_list))
    return point_list

def convert_position_to_grid(point: list[int], grid_map: np.ndarray) -> tuple[int, int]:
    """
    Snap a floating-point coordinate onto a valid grid cell.

    Considers the 4 integer grid positions surrounding the point and
    randomly picks one that's in-bounds and walkable.

    """
    x1 = math.ceil(point[0])
    x2 = math.floor(point[0])
    y1 = math.ceil(point[1])
    y2 = math.floor(point[1])
    candidates = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    random.shuffle(candidates)

    for pos in candidates:
        in_bounds = 0 < pos[0] < grid_map.shape[1] and 0 < pos[1] < grid_map.shape[0]
        if in_bounds and grid_map[pos[1]][pos[0]] != 0:
            return pos
    return (0, 0)


def cluster(
    points: list[list[int]], grid: np.ndarray, number_of_agent: int = 3, step: int = 1
) -> np.ndarray:
    """
    Cluster points into balanced regions using A*-distance spectral clustering.

    Divides the map into number_of_agent regions so a group of robots can
    explore it with balanced workload. Pairwise distance is computed via A*
    rather than Euclidean distance, so clusters respect actual traversal
    cost (walls, obstacles) rather than straight-line proximity.

    """
    city_position = points
    _, distances = record_distance_cpp(
        city_position, grid, step=step, diagonal_movement=True
    )

    ##### Cluster using Spectral Clustering #####
    # Self-tuning affinity (Zelnik-Manor & Perona): use each point's distance to its 
    # number_of_class-th (tunable) nearest neighbor as a local bandwidth.
    sorted_distance = np.sort(distances, axis=1)
    sigma = sorted_distance[:, number_of_agent]
    similarity = np.exp(-(distances**2) / np.outer(sigma, sigma))

    sc = SpectralClustering(
        n_clusters=number_of_agent,
        assign_labels="discretize",
        affinity="precomputed",
    )
    labels = sc.fit_predict(similarity)

    ##### Cluster using Graph Partitioning (alternative, currently unused) #####
    # adjacency = []
    # eweights = []
    # for i in range(city_num):
    #     neighbors = [j for j in range(city_num) if j != i]
    #     weights = [int(distances[i][j]) for j in neighbors]
    #
    #     adjacency.append(neighbors)
    #     eweights.append(weights)
    # xadj = []
    # adjncy = []
    # eweights_flat = []
    #
    # offset = 0
    # for neighbors, weights in zip(adjacency, eweights):
    #     xadj.append(offset)
    #     adjncy.extend(neighbors)
    #     eweights_flat.extend(weights)
    #     offset += len(neighbors)
    # xadj.append(offset)
    # n_cuts, labels = pymetis.part_graph(
    #     nparts=3,
    #     xadj=xadj,
    #     adjncy=adjncy,
    #     eweights=eweights_flat,
    # )
    # for i, label in enumerate(labels):
    #     print(f"Centroid {i} -> Cluster {label}")

    return labels


def cluster_sample_points(
    sample_points_list: list[list[list[int]]],
    grid: np.ndarray,
    number_of_class: int = 3,
) -> list[list[list[tuple[int, int]]]]:
    """
    Cluster sampled boundary points into per-robot groups, one polygon at a time.

    Flattens the per-polygon sample groups into a single point list plus a
    parallel "which polygon did this point come from" grouping, clusters
    all points by position, then reconciles the clustering so every point
    from the same polygon ends up in the same final cluster (a polygon
    can't be split across two robots).

    """
    # Flatten into a single position list plus a parallel grouping by
    # source polygon: city_class[k] holds the (flat) indices into
    # city_position belonging to the k-th input polygon.
    city_position = []
    city_class = []
    n = 0
    for sample in sample_points_list:
        classify = []
        for point in sample:
            city_position.append((point[0], point[1]))
            classify.append(n)
            n += 1
        if len(classify) > 0:
            city_class.append(classify)

    # Generate an initial clustering based on point-to-point A* distance.
    labels = cluster(city_position, grid, number_of_class)
    # print("labels ", labels)

    # Post-process: a polygon's sample points might get split across
    # multiple cluster labels by the generic clustering step above, but a
    # single polygon must belong to exactly one robot. For any polygon
    # whose points span more than one label, reassign ALL of that
    # polygon's points to whichever label currently has the fewest total
    # points (to keep cluster sizes roughly balanced).
    for group in city_class:
        curr_labels = {labels[i] for i in group}
        if len(curr_labels) > 1:
            curr_count = Counter(labels)
            count_list = [curr_count[i] for i in range(number_of_class)]

            label_count = sum(count_list)
            smallest_label = np.argmin(count_list)
            for label in curr_labels:
                if count_list[label] < label_count:
                    smallest_label = label
                    label_count = count_list[label]

            for idx in group:
                labels[idx] = smallest_label

    # print("changed label: ", labels)
    clustered_points_group = [[] for _ in range(number_of_class)]
    for group in city_class:
        label = labels[group[0]]
        group_positions = [city_position[i] for i in group]
        clustered_points_group[label].append(group_positions)

    return clustered_points_group


def cluster_polygon_center(
    convex_set: list[Polygon], grid: np.ndarray, number_of_agent: int = 3, step: int = 1
) -> tuple[list[list[Polygon]], list[list[list[int]]]]:
    """
    Cluster convex regions (by centroid) into groups for each robot to explore.

    """
    centroids = get_centroids(convex_set, grid)
    labels = cluster(centroids, grid, number_of_agent, step)

    convex_groups = []
    selected_points_group = []
    for i in range(number_of_agent):
        group = [polygon for polygon, label in zip(convex_set, labels) if label == i]
        group_centroid = [c for c, label in zip(centroids, labels) if label == i]
        convex_groups.append(group)
        selected_points_group.append(group_centroid)

    return convex_groups, selected_points_group

