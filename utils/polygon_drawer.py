import cv2
import numpy as np

from typing import Union, Optional
from shapely.geometry import (
    Point,
    LineString,
    Polygon,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
)

Color = Union[int, tuple[int, int, int]]

def draw_shapely(
    image: np.ndarray,
    geometry,
    color: Color = (0, 25, 255),
    zoom_rate: float = 1.0,
    background_color: Optional[Color] = 0,
) -> None:
    if geometry is None or isinstance(geometry, Point):
        return

    if isinstance(geometry, LineString):
        points = list(geometry.coords)
        for start, end in zip(points[:-1], points[1:]):
            draw_line(image, start, end, color, zoom_rate)

    elif isinstance(geometry, Polygon):
        draw_polygon(image, geometry, color, zoom_rate, background_color)

    elif isinstance(geometry, (MultiLineString, MultiPolygon, GeometryCollection)):
        for sub_geometry in geometry.geoms:
            draw_shapely(image, sub_geometry, color, zoom_rate)

    else:
        print(f"Unknown geometry type: {type(geometry)}")


def draw_polygon(
    image: np.ndarray,
    polygon: Polygon,
    color: Color = (0, 25, 255),
    zoom_rate: float = 1.0,
    background_color: Optional[Color] = 0,
    preserve_zero: bool = True,
) -> None:
    """Draw a polygon while preserving zero-valued image pixels."""
    points = np.asarray(polygon.exterior.coords) * zoom_rate
    points = np.round(points).astype(np.int32)

    # # Use a mask to avoid overwriting zero-valued pixels in the map.
    # mask = np.zeros_like(image, dtype=np.uint8)
    # cv2.fillPoly(mask, [points], 1)
    # image[(mask == 1) & (image != 0)] = color
    # Use a 2D mask to support both grayscale and RGB images.
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)

    if preserve_zero:
        if image.ndim == 2:
            nonzero = image != 0
        else:
            nonzero = np.any(image != 0, axis=-1)

        image[(mask == 1) & nonzero] = color
    else:
        image[mask == 1] = color

    # Restore polygon holes with the specified background color.
    if background_color is not None:
        for interior in polygon.interiors:
            hole = np.asarray(interior.coords) * zoom_rate
            hole = np.round(hole).astype(np.int32)
            cv2.fillPoly(
                image,
                [hole],
                background_color,
                lineType=cv2.LINE_8,
            )


def draw_line(
    image: np.ndarray,
    pt1: tuple[float, float],
    pt2: tuple[float, float],
    color: Color = (0, 25, 255),
    zoom_rate: float = 1.0,
    thickness: int = 1,
) -> None:
    x1, y1 = (np.asarray(pt1) * zoom_rate).astype(np.int32)
    x2, y2 = (np.asarray(pt2) * zoom_rate).astype(np.int32)

    cv2.line(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness,
    )


def draw_point(
    image: np.ndarray,
    x: float,
    y: float,
    color: Color = (153, 92, 0),
    thickness: int = -1,
    zoom_rate: float = 1.0,
    radius: int = 0,
) -> None:
    center = np.round(np.array([x, y]) * zoom_rate).astype(np.int32)
    cv2.circle(
        image,
        (int(center[0]), int(center[1])),
        radius,
        color,
        thickness,
    )


def draw_triangle(
    image: np.ndarray,
    x: float,
    y: float,
    color: Color = (0, 255, 255),
    zoom_rate: float = 1.0,
    size: int = 10,
) -> None:
    center = np.round(np.array([x, y]) * zoom_rate).astype(np.int32)
    x, y = int(center[0]), int(center[1])

    points = np.array(
        [
            [x, y - size],        # top
            [x - size, y + size], # bottom left
            [x + size, y + size], # bottom right
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(image, [points], color)
