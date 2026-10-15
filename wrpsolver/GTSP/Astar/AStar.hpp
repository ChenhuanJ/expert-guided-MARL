/*
    Copyright (c) 2015, Damian Barczynski <daan.net@wp.eu>
    Following tool is licensed under the terms and conditions of the ISC license.
    For more information visit https://opensource.org/licenses/ISC.
*/
#ifndef __ASTAR_HPP_8F637DB91972F6C878D41D63F7E7214F__
#define __ASTAR_HPP_8F637DB91972F6C878D41D63F7E7214F__
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <functional>
#include <unordered_map>
#include <utility>

namespace AStar
{
    struct Vec2i
    {
        long int x, y;

        bool operator == (const Vec2i& coordinates_);
        friend Vec2i operator + (const AStar::Vec2i& left_, const AStar::Vec2i& right_) {
            return{ left_.x + right_.x, left_.y + right_.y };
        }
    };

    using uint = unsigned int;
    using HeuristicFunction = std::function<uint(Vec2i, Vec2i)>;
    using CoordinateList = std::vector<Vec2i>;

    struct Node
    {
        float G, H;
        Vec2i coordinates;
        Node *parent;

        Node(Vec2i coord_, Node *parent_ = nullptr);
        float getScore();
    };

    using NodeSet = std::unordered_map<int,Node*>;

    class Generator
    {
        bool detectCollision(Vec2i coordinates_);
        Node* findNodeOnList(NodeSet& nodes_, Vec2i coordinates_);
        void releaseNodes(NodeSet& nodes_);

    public:
        Generator();
        void setGrid(pybind11::detail::unchecked_reference<int, 2>* _pGrid);
        void setStep(int _step);
        void setWorldSize(Vec2i worldSize_);
        void setDiagonalMovement(bool enable_);
        void setHeuristic(HeuristicFunction heuristic_);
        std::pair<float, std::vector<std::vector<int>>*> findPath(Vec2i source_, Vec2i target_);
        pybind11::detail::unchecked_reference<int, 2>* _cityPos;


    private:
        HeuristicFunction heuristic;
        CoordinateList direction;
        Vec2i worldSize;
        uint directions;
        int step;
        pybind11::detail::unchecked_reference<int, 2>* pGrid;
    };

    class Heuristic
    {
        static Vec2i getDelta(Vec2i source_, Vec2i target_);

    public:
        static float manhattan(Vec2i source_, Vec2i target_);
        static float euclidean(Vec2i source_, Vec2i target_);
        static float octagonal(Vec2i source_, Vec2i target_);
    };
}

#endif // __ASTAR_HPP_8F637DB91972F6C878D41D63F7E7214F__
