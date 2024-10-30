import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="chage testbed path to your testbed's local path (should be absolute path, e.g., xxx/xxx/testbed)")
    parser.add_argument("local_path", type=str, help="testbed's local path (should be absolute path, e.g., xxx/xxx/testbed)")

    args = parser.parse_args()
    # print(f"Argument 1: {args.local_path}")
    local_path = args.local_path

    setup_map_path = 'SWE-bench/setup_result/setup_map.json'
    with open(setup_map_path, 'r') as f:
        setup_map = json.load(f)
    # change repo_path to local path
    for key, val in setup_map.items():
        val['repo_path'] = os.path.join(local_path, key)
    with open(setup_map_path, 'w') as f:
        json.dump(setup_map, f, indent=4)


if __name__ == '__main__':
    main()