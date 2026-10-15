#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <mutex>
#include "AStar.hpp"
#include "thread_pool.hpp"

using namespace std::placeholders;

bool AStar::Vec2i::operator == (const Vec2i& coordinates_)
{
    return (x == coordinates_.x && y == coordinates_.y);
}

AStar::Node::Node(Vec2i coordinates_, Node *parent_)
{
    parent = parent_;
    coordinates = coordinates_;
    G = H = 0;
}

float AStar::Node::getScore()
{
    return G + H;
}

AStar::Generator::Generator()
{
    setHeuristic(&Heuristic::manhattan);

}

void AStar::Generator::setWorldSize(Vec2i worldSize_)
{
    worldSize = worldSize_;
}

void AStar::Generator::setDiagonalMovement(bool enable_)
{
    directions = (enable_ ? 8 : 4);
}

void AStar::Generator::setHeuristic(HeuristicFunction heuristic_)
{
    heuristic = std::bind(heuristic_, _1, _2);
}

std::pair<float, std::vector<std::vector<int>>*> AStar::Generator::findPath(Vec2i source_, Vec2i target_)
{
    auto res = new std::vector<std::vector<int>>();
    if (detectCollision(source_) || detectCollision(target_)){
        std::cout << "AStar::Generator::findPath input invalid !" << std::endl;
        return {123456, res};
    }
    Node *current = nullptr;
    NodeSet openSet, closedSet;
    openSet.reserve(10000);
    closedSet.reserve(10000);
    openSet.emplace(source_.x*10000+source_.y,new Node(source_));

    while (!openSet.empty()) {
        auto current_it = openSet.begin();
        current = current_it->second;

        for (auto it = openSet.begin(); it != openSet.end(); it++) {
            auto node = it->second;
            if (node->getScore() < current->getScore() ||
               (node->getScore() == current->getScore() && node->H < current->H)) {
                current = node;
                current_it = it;
            }
        }

        if (current->coordinates == target_) {
            break;
        }

        // closedSet.push_back(current);
        closedSet.emplace(current->coordinates.x*10000+current->coordinates.y,current);
        openSet.erase(current_it->second->coordinates.x*10000+current_it->second->coordinates.y);

        for (uint i = 0; i < directions; ++i) {
            Vec2i newCoordinates(current->coordinates + direction[i]);
            if (detectCollision(newCoordinates) ||
                findNodeOnList(closedSet, newCoordinates)) {
                continue;
            }

            float totalCost = current->G + ((i < 4) ? 1 : 1.4);

            Node *successor = findNodeOnList(openSet, newCoordinates);
            if (successor == nullptr) {
                successor = new Node(newCoordinates, current);
                successor->G = totalCost;
                successor->H = heuristic(successor->coordinates, target_);
                // openSet.push_back(successor);
                openSet.emplace(successor->coordinates.x*10000+successor->coordinates.y,successor);
                
            }
            else if (totalCost < successor->G) {
                successor->parent = current;
                successor->G = totalCost;
            }
        }
    }
    if(!(current->coordinates == target_))
        return {245789, res};

    CoordinateList path;
    std::vector<int> temp(2);

    // the total path length is the G cost of current
    float finalCost = current->G;
    while (current != nullptr) {
        // path.push_back(current->coordinates);
        temp[0] = current->coordinates.x;
        temp[1] = current->coordinates.y;
        res->emplace_back(temp);
        current = current->parent;
    }

    releaseNodes(openSet);
    releaseNodes(closedSet);
    return {finalCost, res};
}

AStar::Node* AStar::Generator::findNodeOnList(NodeSet& nodes_, Vec2i coordinates_)
{

    // for (auto node : nodes_) {
    //     if (node->coordinates == coordinates_) {
    //         return node;
    //     }
    // }
    auto iter = nodes_.find(coordinates_.x*10000+coordinates_.y);
    if (iter == nodes_.end()){
        return nullptr;
    }
    return iter->second;
}

void AStar::Generator::releaseNodes(NodeSet& nodes_)
{
    for (auto it = nodes_.begin(); it != nodes_.end();) {
        delete it->second;
        it = nodes_.erase(it);
    }
}

bool AStar::Generator::detectCollision(Vec2i coordinates_)
{
    if (coordinates_.x < 0 || coordinates_.x >= worldSize.x ||
        coordinates_.y < 0 || coordinates_.y >= worldSize.y ||
        (*pGrid)(coordinates_.y,coordinates_.x)==0) {
        return true;
    }
    return false;
}
void AStar::Generator::setGrid(pybind11::detail::unchecked_reference<int, 2>* _pGrid)
{
    pGrid = _pGrid;
}
void AStar::Generator::setStep(int _step)
{
    step = _step;
    if (directions == 4){
        direction = {{ 0, step }, { step, 0 }, { 0, -step }, { -step, 0 },};
    }
    else if (directions == 8)
    {
        direction = {
            { 0, step }, { step, 0 }, { 0, -step }, { -step, 0 },
            { step, step }, { -step, -step }, { step, -step }, { -step, step },
        };
    }
    
}
AStar::Vec2i AStar::Heuristic::getDelta(Vec2i source_, Vec2i target_)
{
    return{ abs(source_.x - target_.x),  abs(source_.y - target_.y) };
}

float AStar::Heuristic::manhattan(Vec2i source_, Vec2i target_)
{
    auto delta = std::move(getDelta(source_, target_));
    return (10 * (delta.x + delta.y));
}

float AStar::Heuristic::euclidean(Vec2i source_, Vec2i target_)
{
    auto delta = std::move(getDelta(source_, target_));
    return (10 * sqrt(pow(delta.x, 2) + pow(delta.y, 2)));
}

float AStar::Heuristic::octagonal(Vec2i source_, Vec2i target_)
{
    auto delta = std::move(getDelta(source_, target_));
    return 10 * (delta.x + delta.y) + (-6) * std::min(delta.x, delta.y);
}
AStar::Generator generator;

using dvector = std::vector<std::vector<int>>;
std::vector<std::vector<dvector*>> * res = nullptr;
std::vector<std::vector<float>>* cost_res = nullptr;

/**
 * Computes (or reuses) the shortest path between city i and city j.
 * 
 * Optimization: path(i -> j) and path(j -> i) are the same route, just
 * traversed in opposite directions and with the same cost. So before running
 * A* we check whether the "opposite" entry res[j][i] has already been
 * filled in by another call; if so we just copy + reverse it instead of
 * recomputing.
 */
void GetSinglePath(int i, int j) {
    dvector* path = nullptr;
    float cost = 0;

    if ((*res)[j][i] != 0) {
        // Opposite direction already computed — reuse it.
        // (*(...)) deep-copies the vector so res[i][j] and res[j][i]
        // own independent buffers (safe to free separately later).
        path = new dvector(*((*res)[j][i]));
        std::reverse(path->begin(), path->end());
        cost = (*cost_res)[j][i];
    } else {
        // Not cached yet — run A* from scratch.
        auto cityPos = *(generator._cityPos);
        AStar::Vec2i start({cityPos(i, 0), cityPos(i, 1)});
        AStar::Vec2i target({cityPos(j, 0), cityPos(j, 1)});
        auto result = generator.findPath(start, target);
        cost = result.first;
        path = result.second;

        std::reverse(path->begin(), path->end());
    }

    (*res)[i][j] = path;
    (*cost_res)[i][j] = cost;
}

namespace py = pybind11;

PYBIND11_MODULE(Astar_cpp, m) {

    /**
     * findPathBetween2Points(grid, startP, endP, isDiagonal, step)
     *
     * Thin wrapper exposing a single-pair A* query to Python. Only the cost
     * is returned; the actual path (the second element of the pair) is
     * discarded via the `_` binding.
     */
    m.def("findPathBetween2Points", [](py::array_t<int> grid_,
                                        std::vector<int> startP,
                                        std::vector<int> endP,
                                        bool isDiagonal,
                                        int step) {
        // unchecked<2>() skips bounds-checking on element access for speed;
        // caller is trusted to pass a real 2D array.
        auto grid = grid_.unchecked<2>();
        auto pGrid = &grid;

        AStar::Vec2i start({startP[0], startP[1]});
        AStar::Vec2i end({endP[0], endP[1]});

        // NOTE: generator is a shared/global instance (declared elsewhere).
        // Every call here mutates its shared state (world size, heuristic,
        // grid pointer, etc.) before running the query, so this function is
        // NOT safe to call concurrently with itself or with GetPath() —
        // there's no synchronization guarding generator.
        generator.setWorldSize({grid.shape(1), grid.shape(0)});
        generator.setHeuristic(AStar::Heuristic::euclidean);
        generator.setDiagonalMovement(isDiagonal);
        generator.setGrid(pGrid);
        generator.setStep(step);

        // structured binding: we only care about the cost (first), so the
        // path itself (second) is bound to `_` and left unused.
        auto [cost, _] = generator.findPath(start, end);
        return cost;
    });

    /** 
     * GetPath(grid, cityPos, step, diagonalMovement)
     *    
     * Computes all-pairs shortest paths between cityNum cities on grid.
     * Returns (res, cost_res):
     *   res[i][j]      -> pointer to path (dvector*) from city i to city j
     *   cost_res[i][j] -> path cost from city i to city j
     */
    m.def("GetPath", [](py::array_t<int> grid_, py::array_t<int> cityPos_,
                         int step, bool diagonalMovement) {
        auto grid = grid_.unchecked<2>();
        auto pGrid = &grid;
        auto cityPos = cityPos_.unchecked<2>();
        int cityNum = cityPos.shape(0);

        res = new std::vector<std::vector<dvector*>>(cityNum, std::vector<dvector*>(cityNum, 0));
        cost_res = new std::vector<std::vector<float>>(cityNum, std::vector<float>(cityNum, 0.0));

        generator.setWorldSize({grid.shape(1), grid.shape(0)});
        generator.setHeuristic(AStar::Heuristic::euclidean);
        generator.setDiagonalMovement(diagonalMovement);
        generator.setGrid(pGrid);
        generator.setStep(step);
        generator._cityPos = &cityPos;

        BS::thread_pool pool(16);
        bool useMultiThread = true;
        if (useMultiThread) {
            // Dispatch every (i, j) pair — including both (i,j) and (j,i) —
            // as independent tasks.
            for (py::ssize_t i = 0; i < cityNum; i++) {
                for (py::ssize_t j = 0; j < cityNum; j++) {
                    // `start` computed here is unused — GetSinglePath()
                    // recomputes it internally from generator._cityPos.
                    // Dead code; safe to remove.
                    AStar::Vec2i start({cityPos(i, 0), cityPos(i, 1)});
                    pool.push_task(GetSinglePath, i, j);
                }
            }
            pool.wait_for_tasks();
        } else {
            // Single-threaded fallback. Iteration order guarantees res[j][i]
            // (j < i, computed on an earlier outer-loop pass) exists 
            // before we ever look it up here.
            for (py::ssize_t i = 0; i < cityNum; i++) {
                AStar::Vec2i start({cityPos(i, 0), cityPos(i, 1)});
                for (py::ssize_t j = 0; j < cityNum; j++) {
                    dvector* path = nullptr;
                    float cost = 0;

                    if ((*res)[j][i] != 0) {
                        // Reuse the already-computed opposite-direction path.
                        path = new dvector(*((*res)[j][i]));
                        std::reverse(path->begin(), path->end());
                        cost = (*cost_res)[j][i];
                    } else {
                        AStar::Vec2i end({cityPos(j, 0), cityPos(j, 1)});
                        auto result = generator.findPath(start, end);
                        cost = result.first;
                        path = result.second;
                        // A* returns the path target->start internally;
                        // reverse to get start->target order.
                        std::reverse(path->begin(), path->end());
                    }

                    (*res)[i][j] = path;
                    (*cost_res)[i][j] = cost;
                }
            }
        }

        return py::make_tuple(res, cost_res);
    }, py::return_value_policy::move);
}