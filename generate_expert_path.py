import json
import os
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

MAP_TYPE: str = 'room'
OUTPUT_DIR: str = f'./expert_path/{MAP_TYPE}_path'
INPUT_FILES: list[str] = [f'dataset/{MAP_TYPE}_train.json']
MAX_WORKERS: int = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Populated by run_single_map_subprocess as maps fail; read back in
# __main__ once all futures complete, to report/persist which maps need
# a rerun.
failed_maps: list[str] = []

def run_single_map_subprocess(item: tuple[str, dict]) -> None:
    """
    Solve a single map by delegating to run_single_map.py in a subprocess,
    and write its result to OUTPUT_DIR/{mapId}.json.

    Writes the single map's data to a temp JSON file (since the worker
    script takes a file path, not inline data), invokes the worker, then
    cleans up the temp file regardless of outcome. Already-solved maps
    (output file already exists) are skipped rather than resolved again.

    Args:
        item: (mapId, mapData) pair, as produced by allMaps.items().
            mapData is the map's polygon/hole payload as loaded from the
            input JSON.
    """
    map_id, map_data = item
    map_file = f'temp_map_{map_id}.json'
    output_path = os.path.join(OUTPUT_DIR, f'{map_id}.json')

    if os.path.exists(output_path):
        print(f"Map {map_id} already exists, skipping.")
        return
    
    try:
        with open(map_file, 'w') as f:
            json.dump({map_id: map_data}, f)

        print(f"{map_id}: solver starts")
        subprocess.run(
            [
                sys.executable, 
                'run_single_map.py',
                map_file, 
                map_id, 
                output_path
            ], 
            check=True
        )
        print("Finished current map")

    except subprocess.CalledProcessError as e:
        print(f"Map {map_id} failed with error: {e}")
        failed_maps.append(map_id)

    finally:
        # Always clean up the temp file, whether the subprocess
        # succeeded, failed, or the write itself raised.
        if os.path.exists(map_file):
            os.remove(map_file)


if __name__ == "__main__":
    # Merge all input files into one {mapId: mapData} lookup before
    # dispatching, so maps from multiple files are processed uniformly.
    all_maps: dict[str, dict] = {}
    for input_file in INPUT_FILES:
        with open(input_file, 'r') as f:
            maps = json.load(f)
            all_maps.update(maps)

    items = list(all_maps.items())[:1]  # Adjust for testing or full run

    start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_single_map_subprocess, item) for item in items]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass
    print(f"Running time: {time.time() - start:.2f} seconds")

    if failed_maps:
        # failedFile = f'failed_maps_{MAP_TYPE}.json'
        print(f"Failed maps: {failed_maps}")
        # with open(failedFile, 'w') as f:
            # json.dump(failed_maps, f)
