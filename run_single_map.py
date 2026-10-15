import json
import os
import sys

import psutil
from shapely.geometry import Polygon
from shapely.validation import make_valid

from utils import save_json
from wrpsolver.main import k_WatchmanRouteProblemSolver
from config import NUM_AGENTS, SENSOR_RANGE, SUCCESS_COVERAGE, STEP

def report_memory() -> None:
    """
    Print current process RSS.
    """
    memoryMb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Memory used: {memoryMb:.2f} MB")

def process_single_map(map_file: str, map_id: str, output_path: str) -> None:
    """
    Load one map entry from a batch map file, solve the k-WRP for it, and
    write the resulting expert path to `outputPath` as JSON.

    Args:
        map_file: path to a JSON file containing a list of map entries,
            each with "polygon" (exterior ring) and "hole" (interior
            rings) keys.
        map_id: index into that list selecting which map to process.
        output_path: destination path for the resulting path JSON.

    Raises:
        RuntimeError: if the solver does not complete successfully
            (invalid polygon, sampling mismatch, etc.) — see
            `k_WatchmanRouteProblemSolver`'s `isSuccess` return value.
    """
    with open(map_file, 'r') as f:
        maps = json.load(f)

    map_data = maps[map_id]

    points = map_data["polygon"]
    holes = map_data["hole"]

    polygon = Polygon(shell=points, holes=holes).simplify(
        0.05, preserve_topology=True
    )

    if not polygon.is_valid:
        polygon = make_valid(polygon).geoms[0]

    (
        _,
        _,
        _,
        _,
        _,
        expert_path,
        is_success,
        _,
    ) = k_WatchmanRouteProblemSolver(
        polygon,
        SUCCESS_COVERAGE,
        SENSOR_RANGE,
        NUM_AGENTS,
        step=STEP,
    )

    if not is_success:
        raise RuntimeError(
            f"Solver did not succeed for map {map_id} in {map_file}"
        )

    holes = [list(ring.coords) for ring in polygon.interiors]
    data = {
        "exterior": list(polygon.exterior.coords),
        "holes": holes,
        "pathList": expert_path,
    }

    save_json(data, output_path)
    report_memory()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <mapFile> <mapId> <outputPath>")
        sys.exit(1)

    map_file: str = sys.argv[1]
    map_id: str = sys.argv[2]
    output_path: str = sys.argv[3]

    try:
        process_single_map(map_file, map_id, output_path)
    except Exception as e:
        print(f"Error processing map {map_id}: {e}")
        sys.exit(1)
