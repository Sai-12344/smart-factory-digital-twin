import numpy as np

class FactoryEnvironment:
    """
    RL environment for Smart Factory Scheduling.
    State = [machine_load, health_score, energy_usage]
    Actions = {0: keep running, 1: cool down, 2: shift load, 3: inspect}
    """

    def __init__(self, num_machines=3):
        self.num_machines = num_machines
        self.reset()

    def reset(self):
        # Initialize machine states randomly
        self.machine_load = np.random.uniform(0.3, 0.9, self.num_machines)
        self.health_score = np.random.uniform(0.4, 1.0, self.num_machines)
        self.energy_usage = np.random.uniform(20, 70, self.num_machines)

        return self._get_state()

    def _get_state(self):
        """Return current factory state."""
        return np.concatenate([
            self.machine_load,
            self.health_score,
            self.energy_usage
        ])

    def step(self, action):
        """
        Apply action:
        0 = keep running
        1 = cool down
        2 = shift load
        3 = inspect (drone)
        """

        reward = 0

        if action == 0:  # keep running
            self.machine_load += np.random.uniform(-0.02, 0.05, self.num_machines)
            reward += np.sum(self.health_score * 2)

        elif action == 1:  # cool down
            self.machine_load -= 0.1
            reward += 5

        elif action == 2:  # shift load
            self.machine_load = np.roll(self.machine_load, 1)
            reward += 3

        elif action == 3:  # inspect
            reward += np.sum(self.health_score)

        # Energy increases slightly with load
        self.energy_usage = 20 + self.machine_load * 80

        # Health decreases with high load
        self.health_score -= np.maximum(0, self.machine_load - 0.7) * 0.05

        # Prevent negative
        self.health_score = np.clip(self.health_score, 0, 1)
        self.machine_load = np.clip(self.machine_load, 0, 1)

        # Determine if episode ends
        done = np.any(self.health_score < 0.1)

        return self._get_state(), reward, done