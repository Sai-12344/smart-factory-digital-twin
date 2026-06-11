import numpy as np
import random
import math


class Machine:
    """
    Industry-style machine simulation.

    Simulates:
    - Wear & tear
    - Stress accumulation
    - Temperature rise
    - Sensor drift
    - Failure probability
    - Maintenance reset effects
    """

    def __init__(self, machine_id):
        self.machine_id = machine_id

        # Core state
        self.age = 0                    # Time cycles
        self.wear = 0.0                 # 0 to 1
        self.health = 1.0               # 1 = perfect, 0 = failure
        self.temperature = 40.0         # Celsius
        self.vibration = 0.05           # G-force
        self.load = random.uniform(0.3, 0.9)

        # Environment
        self.ambient_temp = 28.0        # Celsius
        self.humidity = 0.6             # 0 to 1

        # Failure threshold
        self.failure_prob = 0.0

    def step(self, action="run"):
        """
        One time-step of the machine.

        action:
        - "run"
        - "cool"
        - "maintain"
        - "idle"
        """

        self.age += 1

        # ==== Action Effects ====
        if action == "run":
            self.load += random.uniform(-0.05, 0.1)
            self.load = min(max(self.load, 0.2), 1.0)

        elif action == "cool":
            self.temperature -= 3
            self.vibration -= 0.002
            self.load *= 0.9

        elif action == "maintain":
            self.wear *= 0.4
            self.temperature -= 5
            self.vibration *= 0.7
            self.health = min(1.0, self.health + 0.3)

        elif action == "idle":
            self.load *= 0.5
            self.temperature -= 1

        # ==== Wear Progression ====
        wear_increase = (self.load * 0.02) + (self.temperature - self.ambient_temp) * 0.001
        wear_increase *= random.uniform(0.8, 1.2)

        self.wear += wear_increase
        self.wear = min(self.wear, 1.0)

        # ==== Temperature Dynamics ====
        self.temperature += (self.load * 8 + self.wear * 4) * 0.1
        self.temperature += (self.ambient_temp - self.temperature) * 0.02  # cooling towards ambient

        # ==== Vibration ====
        self.vibration += (self.wear * 0.005) + random.uniform(-0.001, 0.002)

        # ==== Health ====
        self.health = 1.0 - self.wear
        self.health = max(0.0, self.health)

        # ==== Failure Probability ====
        self.failure_prob = self.wear * 0.6 + (self.temperature - 60) * 0.01
        self.failure_prob = max(0.0, min(self.failure_prob, 1.0))

        # ==== Sensors Output ====
        return {
            "id": self.machine_id,
            "age": self.age,
            "temp": round(self.temperature, 2),
            "vibration": round(self.vibration, 3),
            "load": round(self.load, 3),
            "wear": round(self.wear, 3),
            "health": round(self.health, 3),
            "failure_prob": round(self.failure_prob, 3)
        }
    def get_state(self):
        return {
            "id": self.machine_id,
            "age": self.age,
            "temp": round(self.temperature, 2),
            "vibration": round(self.vibration, 3),
            "load": round(self.load, 3),
            "wear": round(self.wear, 3),
            "health": round(self.health, 3),
            "failure_prob": round(self.failure_prob, 3)
        }

    def simulate_step(self, action):
        return self.step(action)

class FactorySim:
    """ Simulates an entire factory with multiple machines. """

    def __init__(self, num_machines=10):
        self.machines = [Machine(i) for i in range(num_machines)]

    def reset(self):
    
        num_machines = len(self.machines)
        self.machines = [Machine(i) for i in range(num_machines)]

    def step(self, actions=None):
        """
        actions: list of actions for each machine
        If None, all machines run normally.
        """
        if actions is None:
            actions = ["run"] * len(self.machines)

        outputs = []
        for i, machine in enumerate(self.machines):
            outputs.append(machine.step(actions[i]))

        return outputs

