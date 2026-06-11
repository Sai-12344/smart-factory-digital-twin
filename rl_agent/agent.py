import numpy as np
import pickle

class RLAgent:
    """
    Simple Q-learning agent for Smart Factory environment.
    Learns best actions to optimize maintenance & energy usage.
    """

    def __init__(self, state_size, action_size, lr=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size

        # Q-table
        self.q_table = np.zeros((50000, action_size))

        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_state_index(self, state):
        """
        Convert continuous state vectors to a bucket index for Q-table.
        """
        # Normalize state
        norm = (state * 100).astype(int)
        idx = np.sum(norm * np.arange(1, len(norm) + 1))
        idx = idx % 50000
        return idx

    def choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)   # explore
        state_idx = self.get_state_index(state)
        return np.argmax(self.q_table[state_idx])         # exploit

    def learn(self, state, action, reward, next_state):
        """
        Update Q-table using the Q-learning formula.
        """
        state_idx = self.get_state_index(state)
        next_idx = self.get_state_index(next_state)

        target = reward + self.gamma * np.max(self.q_table[next_idx])
        self.q_table[state_idx, action] = (1 - self.lr) * self.q_table[state_idx, action] + self.lr * target

        # Decay epsilon
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    def save(self, path="../models/rl_agent_model.pkl"):
        """Save RL agent model."""
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="../models/rl_agent_model.pkl"):
        """Load RL agent model."""
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)