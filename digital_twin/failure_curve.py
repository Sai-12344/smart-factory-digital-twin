import numpy as np
import random
import math


class FailureCurve:
    """
    Machine physics model for realistic wear/failure progression.
    Used to generate RUL targets and stress curves.
    """

    def __init__(self, mode="exponential"):
        """
        mode options:
        - 'exponential' (common for engines/bearings)
        - 'linear'
        - 'random_shock'
        - 'hybrid'
        """
        self.mode = mode

    def compute_wear(self, cycle, load, temperature):
        """
        Computes wear based on:
        - machine cycle number
        - machine load factor (0–1)
        - machine temperature
        """

        if self.mode == "exponential":
            # Failure accelerates with age
            wear = (1 - math.exp(-0.005 * cycle))  
            wear *= (0.4 + load * 0.6)
            wear *= (1 + max(0, temperature - 60) * 0.01)

        elif self.mode == "linear":
            wear = cycle * 0.001
            wear *= (0.3 + load * 0.7)

        elif self.mode == "random_shock":
            wear = cycle * 0.0005
            wear += random.uniform(0, 0.03) if random.random() < 0.01 else 0

        elif self.mode == "hybrid":
            base = (1 - math.exp(-0.004 * cycle))
            shock = random.uniform(0, 0.05) if random.random() < 0.02 else 0
            temp_factor = 1 + max(0, temperature - 55) * 0.012
            wear = base * temp_factor + shock

        else:
            wear = cycle * 0.001

        return min(wear, 1.0)

    def remaining_life(self, wear):
        """
        Converts wear 0–1 to RUL.
        More realistic than direct linear mapping.
        """
        if wear >= 1:
            return 0

        # Non-linear RUL drop
        return int(200 * (1 - wear)**1.5)

    def generate_failure_curve(self, total_cycles=300):
        """
        Generates a synthetic wear curve for an entire machine.
        """
        wear_list = []
        for t in range(total_cycles):
            wear = self.compute_wear(t, load=random.uniform(0.3, 0.9), temperature=random.uniform(40, 90))
            wear_list.append(wear)
        return wear_list