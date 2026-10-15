#!/bin/bash
set -e

# Always run relative to the project root (parent of scripts/), no matter where it's invoked from
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "Initialize A* algorithm..."
(
  cd wrpsolver/GTSP/Astar
  c++ -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) AStar.cpp \
      -o Astar_cpp$(python3-config --extension-suffix)
)

echo "Initialize visible polygon compute algorithm..."
(
  cd wrpsolver/MACS
  c++ -Wall -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) compute_visibilty_cpp.cpp \
      -o visibility$(python3-config --extension-suffix) -lgmp -lmpfr
)
