import math
 
import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
 
from .clustering import cluster_polygon_center, cluster_sample_points

if __name__ == "__main__":
    import os
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )

from wrpsolver import MACS, GTSP
from utils import get_start_points
from config import CLOSE_SPAWN_WRP

CLOSE_SPAWN = CLOSE_SPAWN_WRP

def k_WatchmanRouteProblemSolver(
    polygon: Polygon,
    coverage: float,
    sensor_range: float,
    number_of_agent: int = 3,
    iteration: int = 16,
    step: int = 1
):
    """
    This function solves the multi-agent WRP with open end position by dividing
    it into the Art Gallery Problem (AGP) and a multi-agent TSPN problem. The
    multi-agent TSPN problem is solved by first clustering the region then
    solve the single-agent TSPN problem individually. The single agent TSPN
    problem and the AGP are solved by the following paper:
        https://ieeexplore.ieee.org/document/9736678
 
    Args:
        polygon:
            Contains the information of the entire map, the exteriors are
            the boundary of the map and the holes represents the obstacles
            inside the map.
        coverage:
            The float number between (0, 1) represents the desired
            coverage rate.
        sensorRange:
            The float number represents the visibility range, in our
            cases, the robot is equipped with omnidirectional sensor.
        numberOfClass:
            The number of robots to explore the map.
        iteration:
            The max iteration of coverage algorithm.
        step:
            The step in A* algorithm, adjust the step to solve faster.
    Return:
        convexSet: the convex polygons produced by the AGP coverage solver.
        centroids: the centroids of each convex polygon, aligned with
            clusteredPolygon ordering.
        samplePointsGroup: sample points generated for each robot's region.
        selectedPointsGroup: the subset of sample points chosen by the GTSP
            solver for each robot.
        distanceList: total route distance for each robot.
        pathGroup: the actual A* path for each robot.
        isSuccess: whether the whole pipeline completed without error.
        initPosition: the starting position of each robot,
            aligned with clusteredPolygon / centroids ordering.
    """
    sensor_range = sensor_range / 2
    convexSet = []
    samplePointsGroup = []
    selectedPointsGroup = []
    distanceList = []
    pathGroup = []
    isSuccess = True
    initPosition = None
    centroids = None

    # stop if polygon is not valid
    if not isinstance(polygon.buffer(-1, join_style=2), Polygon):
        isSuccess = False
        print("Polygon is not valid")
        return (
            convexSet,
            centroids,
            samplePointsGroup,
            selectedPointsGroup,
            distanceList,
            pathGroup,
            isSuccess,
            initPosition
        )
    
    # print("Solving the coverage problem")
    convexSet = MACS.polygon_cover(polygon, sensor_range, coverage, iteration)
    # print("Coverage problem solved")
    gridMap: NDArray[np.uint8] = polygon2Grid(polygon.buffer(-1, join_style=2), 255)
    initPosition = get_start_points(polygon, CLOSE_SPAWN, number_of_agent)
    # print("init position: ", initPosition)
 
    ##### Cluster by the polygon's center #####
    # print("Solving the clustering problem")
    clusteredPolygon, centroids = cluster_polygon_center(
        convexSet, gridMap, number_of_agent, step
    )
    # print("Clustering problem solved")
 
    # Build the cost matrix: cost_matrix[i, j] = distance from start point i
    # to the nearest centroid of region j.
    cost_matrix = np.zeros((number_of_agent, number_of_agent))
    for i in range(number_of_agent):
        startPoint = initPosition[i]
        for j in range(number_of_agent):
            min_d = 10000  # arbitrary large number
            for cent in centroids[j]:
                cost = GTSP.compute_distance(startPoint, cent, gridMap, step)
                if cost < min_d:
                    min_d = cost
            cost_matrix[i, j] = min_d
 
    # Assign each start point to a region so that total travel distance is minimized. 
    start_indices, region_indices = linear_sum_assignment(cost_matrix)
    best_perm = [0] * number_of_agent
    for start_idx, region_idx in zip(start_indices, region_indices):
        best_perm[region_idx] = start_idx
 
    initPosition = [initPosition[idx] for idx in best_perm]
 
    for idx, convexGroup in enumerate(clusteredPolygon):
        # print(f"Robot{idx} sampling start")
        samplePoints = GTSP.get_sample(
            convexGroup,
            polygon,
            sensor_range/2,
            gridMap,
            step=step
        )
        if len(convexGroup) != len(samplePoints):
            isSuccess = False
            print("Sample points not correct")
            return (
                convexSet,
                centroids,
                samplePointsGroup,
                selectedPointsGroup,
                distanceList,
                pathGroup,
                isSuccess,
                initPosition
            )
        samplePoints.insert(0, [initPosition[idx]])

        samplePointsGroup.append(samplePoints)
        gtspCase = GTSP.post_processing(samplePoints)
        # print(f"Robot{idx} sampling end")
 
        # print(f"Robot{idx} GTSP solver start")
        selectedPoints, distance, path = GTSP.get_trace_glns(gtspCase, gridMap, step)
        # print(f"Robot{idx} GTSP solved")
        selectedPointsGroup.append(selectedPoints)
        distanceList.append(distance)
        pathGroup.append(path)
 
    ##### Cluster the sample points #####
    # Alternative approach: instead of clustering the convex regions first
    # and sampling per-region (the approach used above), sample the whole
    # convex set at once and then cluster the resulting sample points.
    # This was tried and works, but performs worse than clustering the
    # polygons first, so it's disabled for now. Kept here in case it's
    # worth revisiting/benchmarking again later (uses cluster_sample_points
    # from .clustering).
    # print(f"Sampling start")
    # samplePoints= GTSP.GetSample(convexSet,
    #                              polygon,
    #                              dSample=sensorRange/2,
    #                              gridMap=gridMap,
    #                              step=step)
    # if len(convexSet) != len(samplePoints):
    #     isSuccess = False
    #     print("Sample points not correct")
    #     return (
    #         convexSet,
    #         centroids,
    #         samplePointsGroup,
    #         selectedPointsGroup,
    #         distanceList,
    #         pathGroup,
    #         isSuccess,
    #         initPosition
    #     )
    # print(f"Sampling ends")
    # samplePointsGroup = cluster_sample_points(samplePoints, gridMap)
    # print(samplePointsGroup)
    # for idx, samplePoints in enumerate(samplePointsGroup):
    #     gtspCase = GTSP.postProcessing(samplePoints)
    #     print(f"Robot{idx} GTSP solver start")
    #     selectedPoints, distance, path = GTSP.GetTraceGLNS(gtspCase,
    #                                                        gridMap,
    #                                                        step)
    #     print(f"Robot{idx} GTSP solved")
    #     selectedPointsGroup.append(selectedPoints)
    #     distanceList.append(distance)
    #     pathGroup.append(path)
    return (
        convexSet,
        centroids,
        samplePointsGroup,
        selectedPointsGroup,
        distanceList,
        pathGroup,
        isSuccess,
        initPosition
    )


def polygon2Grid(
    polygon: Polygon, color: int
) -> NDArray[np.uint8]:
    """
    Rasterizes a shapely Polygon (with optional interior holes) into a 2D
    grid map. The exterior is filled with `color`, and each interior hole
    is carved back out (filled with 0).
    """
    _, _, maxx, maxy = polygon.bounds
    maxx = math.ceil(maxx/10)*10
    maxy = math.ceil(maxy/10)*10
    gridMap = np.zeros((int(maxy), int(maxx)), dtype=np.uint8)

    points = list(polygon.exterior.coords)
    # list -> ndarray
    points = np.array(points)
    points = np.round(points).astype(np.int32)
    gridMap = cv2.fillPoly(gridMap, [points], color) # type: ignore

    holes = [list(interior.coords) for interior in polygon.interiors]
    for i in range(len(holes)):
        hole = np.array(holes[i])
        hole = np.round(hole).astype(np.int32)
        gridMap = cv2.fillPoly(gridMap, [hole], color=0, lineType=8) # type: ignore

    return gridMap
