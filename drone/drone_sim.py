import random
import numpy as np

class DroneSim:
    """
    Simple grid-based drone simulator for factory inspection.
    Grid cells can contain machines (with health scores).
    """

    def __init__(self, width=6, height=4, num_machines=6, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.width = width
        self.height = height
        self.num_machines = min(num_machines, width * height)
        self.reset()

    def reset(self):
        # Place machines randomly on grid and assign health scores (0-1)
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        coords = [(r, c) for r in range(self.height) for c in range(self.width)]
        random.shuffle(coords)
        self.machine_positions = coords[:self.num_machines]
        self.machine_health = {}
        for idx, (r, c) in enumerate(self.machine_positions):
            self.grid[r][c] = f"M{idx}"
            # health between 0.2 and 1.0
            self.machine_health[f"M{idx}"] = round(random.uniform(0.25, 1.0), 3)

        # Drone starts at top-left
        self.drone_pos = [0, 0]
        return self._get_obs()

    def _get_obs(self):
        """Return observation: drone position, machine positions, and health map"""
        return {
            "drone_pos": tuple(self.drone_pos),
            "machines": dict(self.machine_health),
            "machine_positions": list(self.machine_positions)
        }

    def step(self, action):
        """
        action: one of ['up','down','left','right','scan']
        Returns: obs, info
        """
        r, c = self.drone_pos
        if action == "up":
            r = max(0, r - 1)
        elif action == "down":
            r = min(self.height - 1, r + 1)
        elif action == "left":
            c = max(0, c - 1)
        elif action == "right":
            c = min(self.width - 1, c + 1)
        elif action == "scan":
            pass  # scanning doesn't move

        self.drone_pos = [r, c]

        info = {}
        # If scan, check if a machine is at current cell and return health
        cell = self.grid[r][c]
        if action == "scan" and cell is not None:
            info["scanned_machine"] = cell
            info["health"] = self.machine_health[cell]
        else:
            info["scanned_machine"] = None
            info["health"] = None

        return self._get_obs(), info

    def render_ascii(self):
        """Return an ASCII map showing drone (D) and machines (M#) with health markers."""
        lines = []
        for i in range(self.height):
            row_elems = []
            for j in range(self.width):
                if [i, j] == self.drone_pos:
                    row_elems.append(" D ")
                elif self.grid[i][j] is not None:
                    mid = self.grid[i][j]
                    h = self.machine_health[mid]
                    row_elems.append(f"{mid}:{int(h*100):02d}")
                else:
                    row_elems.append(" . ")
            lines.append(" | ".join(row_elems))
        return "\n".join(lines)