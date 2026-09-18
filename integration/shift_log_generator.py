# integration/shift_log_generator.py
# Generates a synthetic shift log of events per Integration Spec.

import json
import os
import random
from typing import Any, Dict, List


def generate_shift_log(n_events: int = 75, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate a shift log with approximately:
    - 60% sensor_reading events
    - 20% disturbance events
    - 20% dock_request events
    """
    random.seed(seed)

    events = []

    # Possible blocks for disturbances
    blocks = ["A", "B", "C", "D", "E"]

    for _ in range(n_events):
        r = random.random()

        if r < 0.60:
            # Sensor reading
            event = {
                "type": "sensor_reading",
                "data": {
                    "vibration": round(random.normalvariate(0.5, 0.2), 3),
                    "current": round(random.normalvariate(1.0, 0.3), 3),
                    "temperature": round(random.normalvariate(30.0, 5.0), 3),
                    "wheel_rpm_mismatch": random.choice([True, False]),
                    "battery_voltage_low": random.choice([True, False])
                }
            }
        elif r < 0.80:
            # Disturbance
            event = {
                "type": "disturbance",
                "after_step": random.randint(0, 5),
                "block": random.choice(blocks)
            }
        else:
            # Dock request
            event = {
                "type": "dock_request",
                "opponent": random.choice(["random", "self"])
            }

        events.append(event)

    # Write to data/shift_log.json
    data_dir = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)

    log_path = os.path.join(data_dir, "shift_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Generated {n_events} events and saved to {log_path}")
    return events


if __name__ == "__main__":
    generate_shift_log()
