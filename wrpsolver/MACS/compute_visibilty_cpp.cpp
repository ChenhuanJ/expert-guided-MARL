#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_naive_point_location.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Triangular_expansion_visibility_2.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <utility>
#include <vector>

using Kernel = CGAL::Exact_predicates_exact_constructions_kernel;
using Point_2 = Kernel::Point_2;
using Segment_2 = Kernel::Segment_2;
using Traits_2 = CGAL::Arr_segment_traits_2<Kernel>;
using Arrangement_2 = CGAL::Arrangement_2<Traits_2>;
using VisibilityAlgorithm = CGAL::Triangular_expansion_visibility_2<Arrangement_2>;

/**
 * Compute the line-of-sight visibility polygon for a watcher in a polygon
 * with optional holes. The boundary rings are converted into a CGAL
 * arrangement, then triangular-expansion visibility is evaluated in the face
 * that contains the watcher.
 * @param point_list Exterior boundary ring, CCW, not closed (no repeated
 *     first/last point).
 * @param watcher Query point; must lie strictly inside the boundary and
 *     outside all holes.
 * @param holes Interior obstacle rings, each not closed.
 * @return Vertices of the visibility polygon, in arbitrary (arrangement
 *     vertex-storage) order — not guaranteed to trace the boundary in
 *     sequence.
 * @throws Undefined behavior if watcher doesn't resolve to a face
 *     (e.g. lies exactly on an edge/vertex) — see *face dereference below.
 */
std::vector<std::pair<double, double>> compute_visibility_cpp(
    std::vector<std::pair<double, double>> point_list,
    std::pair<double, double> watcher,
    std::vector<std::vector<std::pair<double, double>>> holes) {
    std::vector<Point_2> points;
    std::vector<Segment_2> segments;
    std::vector<std::pair<double, double>> result;

    // Convert the exterior ring to closed boundary segments.
    for (const auto& point : point_list) {
        points.emplace_back(point.first, point.second);
    }
    for (std::size_t i = 0; i < points.size(); ++i) {
        segments.emplace_back(points[i], points[(i + 1) % points.size()]);
    }

    // Add each obstacle ring to the same arrangement as the exterior boundary.
    for (const auto& hole : holes) {
        std::vector<Point_2> hole_points;
        for (const auto& point : hole) {
            hole_points.emplace_back(point.first, point.second);
        }
        for (std::size_t i = 0; i < hole_points.size(); ++i) {
            segments.emplace_back(
                hole_points[i], hole_points[(i + 1) % hole_points.size()]);
        }
    }

    Arrangement_2 environment;
    CGAL::insert_non_intersecting_curves(
        environment, segments.begin(), segments.end());

    // Locate the free-space arrangement face that contains the watcher.
    Point_2 query_point(watcher.first, watcher.second);
    CGAL::Arr_naive_point_location<Arrangement_2> point_location(environment);
    CGAL::Arr_point_location_result<Arrangement_2>::Type location =
        point_location.locate(query_point);
    auto* face = boost::get<Arrangement_2::Face_const_handle>(&location);

    // Compute the non-regularized visibility region, then return its vertices
    // as double-precision coordinate pairs for the Python interface.
    Arrangement_2 visibility_output;
    VisibilityAlgorithm visibility(environment);
    visibility.compute_visibility(query_point, *face, visibility_output);

    for (auto vertex = visibility_output.vertices_begin();
         vertex != visibility_output.vertices_end();
         ++vertex) {
        result.emplace_back(
            CGAL::to_double(vertex->point().x()),
            CGAL::to_double(vertex->point().y()));
    }

    return result;
}
PYBIND11_MODULE(visibility, module) {
    module.def(
        "compute_visibility_cpp",
        &compute_visibility_cpp,
        "Compute the visibility polygon from a watcher position.");
}
