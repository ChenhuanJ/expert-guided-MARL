import shapely
from . import visibility

def get_visibility_polygon_cpp(polygon: shapely.Polygon, watcher: shapely.Point) -> shapely.Polygon:
    """
    Compute the visibility polygon seen by a watcher inside a polygon.

    The input polygon is first simplified. If simplification splits it into
    a MultiPolygon or GeometryCollection, the sub-polygon that contains the
    watcher is selected before computing visibility.

    Args:
        polygon: The environment polygon, which may contain holes
            (interiors) representing obstacles.
        watcher: The observer's position. Must lie inside polygon.

    Returns:
        A simplified polygon representing the region visible from watcher.

    Raises:
        ValueError: If, after simplification, no sub-polygon contains
            watcher.
    """
    polygon = polygon.simplify(0.05, preserve_topology=True)

    if isinstance(polygon, (shapely.MultiPolygon, shapely.GeometryCollection)):
        polygon = next(
            (geom for geom in polygon.geoms if geom.contains(watcher)),
            None,
        )
        if polygon is None:
            raise ValueError("watcher is not inside any sub-polygon after simplification")

    point_list = list(polygon.exterior.coords)[:-1]  # drop repeated closing point
    holes = [list(interior.coords)[:-1] for interior in polygon.interiors]

    result = visibility.compute_visibility_cpp(point_list, (watcher.x, watcher.y), holes)

    return shapely.Polygon(result).simplify(0.5, preserve_topology=True)


# def GetVisibilityPolygonCPP(polygon,watcher):
#     polygon = polygon.simplify(0.05,True)
#     if(type(polygon) == shapely.GeometryCollection or type(polygon) == shapely.MultiPolygon):
#         for geom in polygon.geoms:
#             polygon = geom if geom.contains(watcher) else polygon

#     pointList = list(polygon.exterior.coords)
#     pointList.pop()
#     holes = [list(interior.coords) for interior in polygon.interiors]
#     for i in range(len(holes)):
#         holes[i].pop()
#     result = visibility.compute_visibility_cpp(pointList,(watcher.x,watcher.y),holes)
#     return shapely.Polygon(result).simplify(0.5,True)
