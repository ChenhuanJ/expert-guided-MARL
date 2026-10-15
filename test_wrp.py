import json
import math
import os
import shutil

import cv2
import imageio
import numpy as np
from shapely.geometry import Polygon

from dataset import get_polygon
from utils import draw_line, draw_point, draw_polygon, draw_triangle, save_json
from wrpsolver.main import k_WatchmanRouteProblemSolver


ROBOT_COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),   # magenta
]

SELECTED_POINT_COLOR = (99, 48, 105)
INIT_POINT_COLOR = (99, 48, 105)


def _flatten_points(nested):
    """
    Recursively flatten an arbitrarily nested list of points into a flat
    list of (x, y) points.
    """
    if len(nested) == 2 and all(
        isinstance(v, (int, float, np.integer, np.floating))
        for v in nested
    ):
        return [nested]

    flat = []
    for item in nested:
        flat.extend(_flatten_points(item))

    return flat


def test_kwrp(
    polygon: Polygon,
    sensor_range: float = 30,
    coverage_rate: float = 0.97,
    number_of_robot: int = 3,
    step: int = 1,
):
    """Test the multi-robot WRP solver."""
    (
        convex_set,
        centroid_groups,
        sample_points_group,
        selected_points_group,
        distance_list,
        path_group,
        _,
        init_position,
    ) = k_WatchmanRouteProblemSolver(
        polygon,
        coverage_rate,
        sensor_range,
        number_of_robot,
        iteration=16,
        step=step,
    )

    return (
        convex_set,
        centroid_groups,
        sample_points_group,
        selected_points_group,
        distance_list,
        path_group,
        init_position,
    )


def plot_result(
    path_list,
    centroid_groups,
    selected_points,
    convex_polygon_set,
    sample_points,
    save_path,
    seed,
    number_of_robot=3,
):
    """
    Render each stage of the solve to separate PNGs and stitch them into
    a single GIF for visualization.
    """
    assert number_of_robot <= len(ROBOT_COLORS), (
        f"Only {len(ROBOT_COLORS)} robot colors defined, "
        f"got number_of_robot={number_of_robot}"
    )

    zoom_rate = 5
    process_save_path = save_path + "/process_" + str(seed)

    if os.path.exists(process_save_path):
        shutil.rmtree(process_save_path)

    os.mkdir(process_save_path)
    images = []

    # Draw initial polygon.
    min_x, min_y, max_x, max_y = polygon.bounds
    max_x = math.ceil(max_x / 10) * 10 * zoom_rate
    max_y = math.ceil(max_y / 10) * 10 * zoom_rate

    image = np.zeros(
        (int(max_y), int(max_x), 3),
        dtype=np.uint8,
    )

    obstacle = polygon.boundary.buffer(1, join_style=2)

    draw_polygon(
        image,
        polygon,
        (255, 255, 255),
        zoom_rate=zoom_rate,
        preserve_zero=False,
    )

    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(
        process_save_path + "/init_polygon.png",
        image,
    )

    polygon_image = image.copy()

    # Draw convex polygon set.
    color_list = [
        np.random.randint(0, 255, size=(3,)).tolist()
        for _ in range(len(convex_polygon_set))
    ]

    for i in range(len(convex_polygon_set)):
        convex_polygon = convex_polygon_set[i]
        convex_polygon = convex_polygon.simplify(
            0.05,
            preserve_topology=False,
        )

        draw_polygon(
            image,
            convex_polygon,
            color_list[i],
            zoom_rate=zoom_rate,
            background_color=(0, 0, 0),
        )

        cv2.imwrite(
            process_save_path + "/convex_" + str(i) + ".png",
            image,
        )
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw centroids.
    centroid_image = image.copy()

    for idx, centroid_group in enumerate(centroid_groups):
        color = ROBOT_COLORS[idx]

        for centroid in centroid_group:
            draw_point(
                centroid_image,
                centroid[0],
                centroid[1],
                color,
                radius=5,
                zoom_rate=zoom_rate,
            )

    cv2.imwrite(
        process_save_path + "/centroids.png",
        centroid_image,
    )
    images.append(cv2.cvtColor(centroid_image, cv2.COLOR_BGR2RGB))

    # Draw sample points.
    point_groups_by_robot = (
        sample_points
        if number_of_robot != 1
        else [sample_points]
    )

    for idx, point_set in enumerate(point_groups_by_robot):
        color = ROBOT_COLORS[idx]

        for point in _flatten_points(point_set):
            draw_point(
                image,
                point[0],
                point[1],
                color,
                radius=zoom_rate,
                zoom_rate=zoom_rate,
            )

    cv2.imwrite(
        process_save_path + "/sample.png",
        image,
    )
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw selected points.
    selected_groups = (
        selected_points
        if number_of_robot != 1
        else [selected_points]
    )

    for group in selected_groups:
        for i, point in enumerate(group):
            draw_point(
                image,
                point[0],
                point[1],
                SELECTED_POINT_COLOR,
                radius=zoom_rate,
                zoom_rate=zoom_rate,
            )

    cv2.imwrite(
        process_save_path + "/selected_points_0.png",
        image,
    )
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for point_set in selected_groups:
        for i, point in enumerate(point_set):
            if i == 0:
                # Initial position of each robot is drawn as a triangle.
                draw_triangle(
                    polygon_image,
                    point[0],
                    point[1],
                    INIT_POINT_COLOR,
                    zoom_rate=zoom_rate,
                    size=zoom_rate,
                )
            else:
                draw_point(
                    polygon_image,
                    point[0],
                    point[1],
                    SELECTED_POINT_COLOR,
                    radius=zoom_rate,
                    zoom_rate=zoom_rate,
                )

    cv2.imwrite(
        process_save_path + "/selected_points_1.png",
        polygon_image,
    )
    images.append(cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB))

    for idx, point_set in enumerate(selected_groups):
        color = ROBOT_COLORS[idx]

        for i, point in enumerate(point_set):
            if i == 0:
                draw_triangle(
                    polygon_image,
                    point[0],
                    point[1],
                    color,
                    zoom_rate=zoom_rate,
                    size=zoom_rate,
                )
            else:
                draw_point(
                    polygon_image,
                    point[0],
                    point[1],
                    color,
                    radius=zoom_rate,
                    zoom_rate=zoom_rate,
                )

    cv2.imwrite(
        process_save_path + "/selected_points_2.png",
        polygon_image,
    )
    images.append(cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB))

    # Draw paths.
    if number_of_robot != 1:
        for idx, path in enumerate(path_list):
            color = ROBOT_COLORS[idx]

            for i in range(len(path) - 1):
                draw_line(
                    polygon_image,
                    path[i],
                    path[i + 1],
                    color,
                    zoom_rate=zoom_rate,
                    thickness=int(zoom_rate / 2),
                )
    else:
        for i in range(len(path_list) - 1):
            draw_line(
                polygon_image,
                path_list[i],
                path_list[i + 1],
                color=(0, 0, 255),
                zoom_rate=zoom_rate,
                thickness=int(zoom_rate / 2),
            )

    cv2.imwrite(
        process_save_path + "/path.png",
        polygon_image,
    )
    images.append(cv2.cvtColor(polygon_image, cv2.COLOR_BGR2RGB))

    # Hold the last few frames of the GIF longer than the intermediate
    # steps so the final route is easier to look at.
    duration_list = [100] * len(images)
    duration_list[-1] = 2e3
    duration_list[-2] = 1e3
    duration_list[-3] = 1e3
    duration_list[-4] = 1e3
    duration_list[-5] = 1e3
    duration_list[-6] = 1e3

    imageio.v3.imwrite(
        save_path + "/result_" + str(seed) + ".gif",
        np.stack(images, axis=0),
        duration=duration_list,
        loop=0,
    )


if __name__ == "__main__":

    seed = 106
    count = 1

    # Map type should be room or dungeon.
    map_type = "dungeon"
    sensor_range = 30
    coverage_rate = 0.97
    number_of_robot = 3
    step = 1

    save_folder = (
        os.path.dirname(os.path.realpath(__file__))
        + "/result_wrp/"
        + map_type
        + "/"
    )

    os.makedirs(save_folder, exist_ok=True)

    json_folder = (
        os.path.dirname(os.path.realpath(__file__))
        + "/expert_path/"
        + map_type
        + "_path/"
    )

    os.makedirs(json_folder, exist_ok=True)

    for i in range(count):
        polygon, name = get_polygon(seed + i, map_type)

        json_path = json_folder + name + ".json"
        save_path = save_folder

        print(
            f"map: {map_type}{name} solver starts"
        )

        (
            convex_set,
            centroid_groups,
            sample_points_group,
            selected_points_group,
            distance_list,
            path_group,
            init_position,
        ) = test_kwrp(
            polygon,
            sensor_range,
            coverage_rate,
            number_of_robot,
            step,
        )

        holes = [
            list(ring.coords)
            for ring in polygon.interiors
        ]

        data = {
            "exterior": list(polygon.exterior.coords),
            "holes": holes,
            "pathList": path_group,
        }

        save_json(data, json_path)

        plot_result(
            path_group,
            centroid_groups,
            selected_points_group,
            convex_set,
            sample_points_group,
            save_path,
            seed + i,
            number_of_robot,
        )

        for robot_idx in range(number_of_robot):
            print(
                f"Robot {robot_idx} cost: "
                f"{distance_list[robot_idx]}"
            )

        print("Finished")
