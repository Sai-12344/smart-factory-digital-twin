import numpy as np

def obs_to_heatmap(obs, width, height):
    """
    Convert DroneSim observation to a 2D heatmap of health values.
    Missing cells -> np.nan
    """
    heatmap = np.full((height, width), np.nan, dtype=float)
    for mid, pos in zip(obs["machines"].keys(), obs["machine_positions"]):
        r, c = pos
        heatmap[r, c] = obs["machines"][mid]
    return heatmap

def top_k_weak_machines(obs, k=3):
    """
    Return top-k machines with lowest health (most critical).
    """
    items = list(obs["machines"].items())
    items_sorted = sorted(items, key=lambda x: x[1])  # ascending health
    return items_sorted[:k]