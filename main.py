# main.py
# Thin entrypoint to generate log if missing, and call control_loop.run

import json
import os

from integration.control_loop import run
from integration.shift_log_generator import generate_shift_log


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    log_path = os.path.join(data_dir, "shift_log.json")

    if not os.path.exists(log_path):
        print(f"Shift log not found at {log_path}. Generating one...")
        generate_shift_log()

    with open(log_path, "r") as f:
        shift_log = json.load(f)

    run(shift_log)


if __name__ == "__main__":
    main()
