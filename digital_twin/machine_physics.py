import numpy as np
import random
from .failure_curve import FailureCurve


class MachinePhysics:
    """
    Realistic physics-driven machine model combining:
    - failure curve progression
    - heat buildup
    - vibration stress
    - environmental effects
    - probabilistic failure events
    """

    def __init__(self, machine_id, curve_mode="hybrid"):
        self.machine_id = machine_id
        self.curve = FailureCurve(mode=curve_mode)

        # Internal machine state
        self.cycle = 0
        self.load = random.uniform(0.3, 0.9)
        self.temperature = 40
        self.vibration = 0.02
        self.wear = 0
        self.health = 1.0

        # Environment
        self.ambient_temp = 27
        self.humidity = random.uniform(0.4, 0.75)

    def apply_action(self, action):
        """
        Supported actions:
        - run
        - cool
        - idle
        - maintain
        """

        if action == "run":
            self.load = min(1.0, self.load + random.uniform(-0.05, 0.1))

        elif action == "cool":
            self.temperature -= 4
            self.load *= 0.9

        elif action == "idle":
            self.load *= 0.7
            self.temperature -= 2

        elif action == "maintain":
            # Reset wear partially
            self.wear *= 0.5
            self.temperature -= 6
            self.vibration *= 0.7
            self.health = min(1.0, self.health + 0.4)

    def simulate_step(self, action="run"):
        self.cycle += 1

        # ===== Apply action =====
        self.apply_action(action)

        # ===== Calculate wear with physics model =====
        self.wear = self.curve.compute_wear(
            cycle=self.cycle,
            load=self.load,
            temperature=self.temperature,
        )

        # ===== Temperature dynamics =====
        heat_gain = self.load * 1.5 + self.wear * 1.2
        cooling = (self.temperature - self.ambient_temp) * 0.03
        self.temperature = self.temperature + heat_gain - cooling

        # ===== Vibration changes =====
        self.vibration += (self.wear * 0.008) + random.uniform(-0.002, 0.003)

        # ===== Health =====
        self.health = max(0, 1 - self.wear)

        # ===== Failure probability =====
        base_fail = self.wear * 0.55
        heat_fail = max(0, (self.temperature - 65) * 0.015)
        vibration_fail = max(0, (self.vibration - 0.05) * 0.2)

        failure_probability = min(1.0, base_fail + heat_fail + vibration_fail)

        # ===== RUL (Remaining Useful Life) =====
        rul = self.curve.remaining_life(self.wear)

        # ===== Sensor Output =====
        return {
            "id": self.machine_id,
            "cycle": self.cycle,
            "temp": round(self.temperature, 2),
            "vibration": round(self.vibration, 3),
            "load": round(self.load, 3),
            "wear": round(self.wear, 3),
            "health": round(self.health, 3),
            "failure_prob": round(failure_probability, 3),
            "RUL": rul,
        }