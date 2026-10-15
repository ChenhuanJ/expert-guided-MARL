import json
import math
import os
import sys
from typing import TypedDict

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils import draw_polygon

class PolygonData(TypedDict):
    polygon: list[tuple[float, float]]
    hole: list[list[tuple[float, float]]]

PolygonFile = dict[str, PolygonData]

def get_polygon(
    map_seed: int = 1,
    map_type: str = "dungeon",
) -> tuple[Polygon, str]:
    """
    Load one polygon from a JSON file based on the given seed index.

    `map_seed` is the index of the selected key, not the actual JSON key.
    """
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"{map_type}_test.json",
    )

    with open(file_path, "r") as file:
        data: PolygonFile = json.load(file)

    key = list(data.keys())[map_seed]
    polygon_data = data[key]

    polygon = Polygon(
        shell=polygon_data["polygon"],
        holes=polygon_data["hole"],
    ).simplify(
        0.05,
        preserve_topology=True,
    )

    if not polygon.is_valid:
        polygon = make_valid(polygon).geoms[0]

    if not isinstance(polygon, Polygon):
        raise TypeError("Resulting geometry is not a Polygon")

    return polygon, key


if __name__ == "__main__":
    seed = 8
    polygon, key = get_polygon(seed, 'dungeon')
    print("polygon area: ", polygon.area)
    
    zoom_rate = 2
    _, _, max_x, max_y = polygon.bounds
    maxx = math.ceil(max_x / 10) * 10 * zoom_rate
    maxy = math.ceil(max_y / 10) * 10 * zoom_rate
    image = np.zeros((int(maxy), int(maxx), 3), dtype=np.uint8)

    draw_polygon(image, polygon, (255,255,255), zoom_rate=zoom_rate, background_color=230, preserve_zero=False)
    cv2.imwrite('./polygon.png', image)
