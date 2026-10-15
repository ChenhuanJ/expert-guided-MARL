import random
from multiprocessing import Pool
from typing import Optional

import shapely
from shapely.validation import make_valid
from shapely.ops import split, nearest_points

from .compute_visibility import get_visibility_polygon_cpp
Point2D = tuple[float, float]

def _select_max_intersect_polygon(
    polygon: shapely.Polygon, candidates: list[shapely.Polygon]
) -> shapely.Polygon:
    """Return the candidate polygon with the largest intersection area with polygon."""
    result = max(
        candidates,
        key=lambda p: (
            shapely.set_precision(polygon, 1e-6)
            .intersection(shapely.set_precision(p, 1e-6))
        ).area,
    )
    return result


def select_max_polygon(
    polygons: shapely.geometry.base.BaseGeometry
) -> Optional[shapely.Polygon]:
    """
    Recursively extract the largest-area Polygon from a (possibly nested) geometry.

    Handles Polygon, MultiPolygon, and GeometryCollection inputs, descending
    into nested collections to find the single Polygon with the greatest
    area. Invalid geometries are repaired via `make_valid` before searching.

    Args:
        polygons: A shapely Polygon, MultiPolygon, GeometryCollection,
            LineString, or Point.

    Returns:
        The largest-area shapely.Polygon found. Returns an empty
        shapely.Polygon() if polygons is a LineString or Point (i.e.
        has no area).

    """
    max_polygon: Optional[shapely.Polygon] = None
    polygons = make_valid(polygons)
    if isinstance(polygons, (shapely.LineString, shapely.Point)):
        return shapely.Polygon()
    elif isinstance(polygons, shapely.Polygon):
        return polygons
    elif isinstance(polygons, (shapely.MultiPolygon, shapely.GeometryCollection)):
        for polygon in list(polygons.geoms):
            polygon = select_max_polygon(polygon)
            if polygon is None:
                pass
            elif max_polygon is None or polygon.area > max_polygon.area:
                max_polygon = polygon
    else:
        print("Bad polygon type", type(polygons))
    return max_polygon


def _select_point_from_polygon(polygon: shapely.Polygon) -> shapely.Point:
    """
    Return a random point sampled inside polygon, eroded by 1 unit to stay clear of
    the boundary.

    """
    temp_polygon = select_max_polygon(polygon.buffer(-1))
    minx, miny, maxx, maxy = temp_polygon.bounds
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = shapely.Point(x, y)
        if temp_polygon.contains(p):
            return p


def find_visible_region(
    polygon: shapely.Polygon, watcher: shapely.Point, sensor_range: float = 30
) -> Optional[shapely.Polygon]:
    """
    Compute the region of polygon visible from watcher within sensor_range.

    Intersects a disk of radius sensor_range (+1) centered on watcher
    with polygon, then computes the line-of-sight visibility polygon
    within that intersection.

    """
    if not polygon.contains(watcher):
        print("watcher not in polygon!")
        return None
    sensor_polygon = watcher.buffer(sensor_range + 1, cap_style=1)  # sensor range
    sensor_visibility = sensor_polygon.intersection(polygon)  # sensor range within polygon
    visible_polygon = get_visibility_polygon_cpp(sensor_visibility, watcher)  # line of sight
    return visible_polygon.simplify(0.05, preserve_topology=True)


def _get_kernel(
    polygon: shapely.Polygon, watcher: shapely.Point
) -> tuple[shapely.Polygon, list[Point2D]]:
    """
    Compute the kernel (star-shaped visibility core) of polygon and its reflex vertices.

    For each reflex vertex, splits the polygon along the ray extending its
    adjacent edge and keeps the part containing `watcher`, progressively
    shrinking toward the polygon's kernel.

    """
    kernel = polygon
    reflex_vertexes: list[Point2D] = []

    vertexes = list(polygon.exterior.coords)
    vertexes.pop()
    ccw = shapely.LinearRing(vertexes).is_ccw
    vertex_num = len(vertexes)
    for i in range(vertex_num):
        vertex = vertexes[i % vertex_num]
        vertex_left = vertexes[(i - 1) % vertex_num]
        vertex_right = vertexes[(i + 1) % vertex_num]
        is_convex = (
            (vertex[0] - vertex_left[0]) * (vertex_right[1] - vertex_left[1])
            - (vertex_right[0] - vertex_left[0]) * (vertex[1] - vertex_left[1])
        ) < 0
        if is_convex ^ (not ccw):
            reflex_vertexes.append(vertex)
            ray_line = _get_extrapolated_line(vertex, vertex_right)
            split_polygons = split(kernel, ray_line)
            for geom in split_polygons.geoms:
                polygon = geom if geom.contains(watcher) else polygon

    return kernel.simplify(0.05, True), reflex_vertexes


def _get_extrapolated_line(watcher: Point2D, vertex: Point2D) -> shapely.LineString:
    """
    Return a very long line (as a LineString) passing through watcher and vertex, 
    extended far past both.
    
    """
    x_gap = vertex[0] - watcher[0]
    y_gap = vertex[1] - watcher[1]
    rate = 1e6
    extend_point1 = (watcher[0] + x_gap * rate, watcher[1] + y_gap * rate)
    extend_point2 = (watcher[0] - x_gap * rate, watcher[1] - y_gap * rate)
    return shapely.LineString([extend_point1, extend_point2])


def _get_single_reflex_chord(
    vertex: Point2D, kernel: shapely.Polygon
) -> shapely.LineString:
    """
    Compute the chord through a reflex `vertex` used to split off a convex 
    region relative to kernel.

    If vertex lies on`kernel's boundary, the chord slope is the average
    of the slopes of its two adjacent kernel edges. Otherwise, the chord is
    derived from the nearest point on kernel to vertex.

    """
    slope = None
    kernel_vertexes = list(kernel.exterior.coords)
    kernel_vertexes.pop()

    if vertex in list(kernel_vertexes):  # if reflex point is on kernel
        kernel_vertexes_num = len(kernel_vertexes)
        pos = kernel_vertexes.index(vertex)
        pos_left = kernel_vertexes[(pos - 1) % kernel_vertexes_num]
        pos_right = kernel_vertexes[(pos + 1) % kernel_vertexes_num]
        # the slope of the chord should be the average of the slope of the adjacent edges of the reflex point
        if abs(pos_left[0] - vertex[0]) < 1e-2 or abs(pos_right[0] - vertex[0]) < 1e-2:
            extend_point = (vertex[0], vertex[1] + 1e6)
        else:
            slope1 = (pos_right[1] - vertex[1]) / (pos_right[0] - vertex[0])
            slope2 = (pos_left[1] - vertex[1]) / (pos_left[0] - vertex[0])
            slope = (slope1 + slope2) / 2
            extend_point = (vertex[0] + 1, vertex[1] + slope)
    else:
        point = shapely.Point(vertex)  # if reflex point is not on kernel
        nearest_point = (nearest_points(point, kernel))[1]
        if abs(point.y - nearest_point.y) < 1e-2:
            extend_point = (vertex[0], vertex[1] + 1e6)
        else:
            slope = (nearest_point.x - point.x) / (point.y - nearest_point.y)
            extend_point = (vertex[0] + 1, vertex[1] + slope)
    return _get_extrapolated_line(vertex, extend_point)


def _get_split_polygon(
    chord: shapely.LineString, visible_polygon: shapely.Polygon, watcher: shapely.Point
) -> shapely.Polygon:
    """
    Split visible_polygon along chord and return the resulting 
    piece that contains watcher.
    
    """
    result = shapely.Polygon()
    for polygon in list(split(visible_polygon, chord).geoms):
        result = polygon if polygon.contains(watcher) else result

    if isinstance(result, (shapely.GeometryCollection, shapely.MultiPolygon)):
        for geom in result.geoms:
            result = geom if (geom.contains(watcher) and isinstance(geom, shapely.Polygon)) else result
    return result


def _maximally_covering_convex_polygon(
    args: tuple[shapely.Polygon, shapely.Polygon, shapely.Point, float]
) -> shapely.Polygon:
    """
    Compute a maximal convex polygon (containing watcher) that best covers 
    the uncovered region.

    Implements the core per-seed-point step of the coverage algorithm: finds
    the visible region and its kernel from watcher, then iteratively
    trims it using extremal/reflex chords, keeping whichever candidate
    piece overlaps uncovered_polygon the most.

    """
    uncovered_polygon = args[0]
    initial_polygon = args[1]
    watcher = args[2]
    sensor_range = args[3]

    visible_polygon = find_visible_region(initial_polygon, watcher, sensor_range)
    kernel_polygon, reflex_vertexes = _get_kernel(visible_polygon, watcher)
    reflex_vertexes.sort(
        key=lambda watcher: shapely.distance(kernel_polygon, shapely.Point(watcher))
    )

    # initialization for the first iteration
    reflex_vertex_num = len(reflex_vertexes)
    polygon = visible_polygon
    for i in range(reflex_vertex_num):

        polygon_vertexes = list(polygon.exterior.coords)
        polygon_vertexes.pop()
        polygon_vertex_num = len(polygon_vertexes)
        reflex_vertex1 = reflex_vertexes[i]
        reflex_vertex2 = reflex_vertexes[(i + 1) % reflex_vertex_num]

        if reflex_vertex1 not in polygon_vertexes:
            continue

        r1_pos = polygon_vertexes.index(reflex_vertex1)
        r1_left = polygon_vertexes[(r1_pos - 1) % polygon_vertex_num]
        r1_right = polygon_vertexes[(r1_pos + 1) % polygon_vertex_num]

        # extremal chords
        chord = _get_extrapolated_line(reflex_vertex1, r1_left)
        e_polygon1 = _get_split_polygon(chord, polygon, watcher)
        chord = _get_extrapolated_line(reflex_vertex1, r1_right)
        e_polygon2 = _get_split_polygon(chord, polygon, watcher)

        # two reflex chord
        if len(reflex_vertexes) > 1:
            chord = _get_extrapolated_line(reflex_vertex1, reflex_vertex2)
            t_polygon = _get_split_polygon(chord, polygon, watcher)
        else:
            t_polygon = shapely.Polygon()  # area = 0

        # single reflex chord
        chord = _get_single_reflex_chord(reflex_vertex1, kernel_polygon)
        s_polygon = _get_split_polygon(chord, polygon, watcher)
        polygon = _select_max_intersect_polygon(
            uncovered_polygon, [e_polygon1, e_polygon2, t_polygon, s_polygon]
        )
    return polygon


def polygon_cover(
    polygon: shapely.Polygon,
    sensor_range: float,
    coverage: float,
    iterations: int = 16,
) -> list[shapely.Polygon]:
    """
    Cover polygon with a set of convex-ish polygons up to a target coverage fraction.

    Repeatedly picks a random point in the uncovered area, grows the
    maximal covering convex polygon from it (refined over several random
    seed points in parallel), and subtracts the best result from the
    uncovered region. Stops once the covered fraction of polygon reaches
    coverage.

    Args:
        polygon: The shapely.Polygon environment to cover.
        sensor_range: Maximum sensing/visibility distance used when growing
            each covering polygon.
        coverage: Target fraction (0-1) of polygon's area that must be
            covered before stopping.
        iterations: Number of random seed points sampled per outer
            iteration (also used as the multiprocessing pool size).

    Returns:
        A list of shapely.Polygon pieces whose union covers at least
        coverage of polygon's area.

    """
    # initialization
    polygon_cover_list: list[shapely.Polygon] = []
    uncover_polygon = shapely.Polygon(polygon)
    pool = Pool(iterations)  # multiprocessing

    while (uncover_polygon.area / polygon.area) > (1 - coverage):

        point = _select_point_from_polygon(uncover_polygon)
        best_c = _maximally_covering_convex_polygon(
            (uncover_polygon, polygon, point, sensor_range)
        )  # C denotes the convex polygon

        # iteration begins
        point_list = []
        for _ in range(iterations):
            point_list.append(_select_point_from_polygon(best_c))
        r_list = pool.map(
            _maximally_covering_convex_polygon,
            [(uncover_polygon, polygon, point, sensor_range) for point in point_list],
        )
        best_c = _select_max_intersect_polygon(uncover_polygon, r_list)
        polygon_cover_list.append(best_c)
        uncover_polygon = uncover_polygon.difference(best_c)
        uncover_polygon = make_valid(uncover_polygon.simplify(0.05, True))
        if isinstance(uncover_polygon, shapely.GeometryCollection):
            uncover_polygon = shapely.MultiPolygon(
                [geom for geom in uncover_polygon.geoms if isinstance(geom, shapely.Polygon)]
            )
    pool.close()
    pool.join()

    return polygon_cover_list