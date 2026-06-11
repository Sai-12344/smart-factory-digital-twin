import numpy as np
import pandas as pd

def calculate_energy_efficiency(power_usage, production_output):
    """
    Calculate energy efficiency score.
    Higher = better.
    """
    if production_output == 0:
        return 0.0
    efficiency = production_output / (power_usage + 1e-6)
    return round(float(efficiency), 4)


def detect_idle_machines(sensor_row, idle_threshold=0.1):
    """
    Detect machines that are consuming power but not producing output.
    sensor_row -> dict of sensor values for a machine.
    """
    power = sensor_row.get("power", 0)
    load = sensor_row.get("load", 0)

    is_idle = power > 5 and load < idle_threshold
    return bool(is_idle)


def optimize_energy_usage(sensor_data_list):
    """
    Suggest power optimization actions based on recent sensor data.
    sensor_data_list: list of sensor dicts (recent readings)
    """
    actions = []
    for idx, row in enumerate(sensor_data_list):
        power = row.get("power", 0)
        load = row.get("load", 0)

        if power > 50 and load < 0.2:
            actions.append((idx, "Reduce load or idle the machine"))

        if power > 70:
            actions.append((idx, "Shift load to another machine"))

        if load > 0.9:
            actions.append((idx, "Machine overloaded: redistribute tasks"))

    return actions


def compute_total_energy_usage(sensor_data_list):
    """
    Compute total energy consumed by all machines.
    """
    total = sum(row.get("power", 0) for row in sensor_data_list)
    return float(total)