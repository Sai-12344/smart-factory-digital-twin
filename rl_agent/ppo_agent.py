import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# ---------------------------------------------------------
# PPO NETWORK (Actor + Critic)
# ---------------------------------------------------------
class PPOActorCritic(Model):
    def __init__(self, action_dim):
        super().__init__()
        self.fc1 = Dense(128, activation="relu")
        self.fc2 = Dense(128, activation="relu")

        # Actor output (policy)
        self.actor_out = Dense(action_dim, activation="softmax")

        # Critic output (value estimate)
        self.critic_out = Dense(1, activation="linear")

    def call(self, x):
        h = self.fc1(x)
        h = self.fc2(h)

        policy = self.actor_out(h)
        value = self.critic_out(h)

        return policy, value


# ---------------------------------------------------------
# PPO AGENT
# ---------------------------------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, clip_ratio=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        # Actor-Critic model
        self.model = PPOActorCritic(action_dim)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    # -----------------------------------------------------
    # Choose action
    # -----------------------------------------------------
    def act(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        policy, _ = self.model(state)
        policy = policy.numpy()[0]
        action = np.random.choice(len(policy), p=policy)
        return action, policy

    # -----------------------------------------------------
    # Compute advantages
    # -----------------------------------------------------
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        gamma = self.gamma
        lam = 0.95  # GAE parameter

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * gae * (1 - dones[t])
            advantages.insert(0, gae)

        return np.array(advantages)

    # -----------------------------------------------------
    # PPO TRAINING STEP
    # -----------------------------------------------------
    def train(self, states, actions, rewards, next_states, dones, old_policies):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = np.array(actions)

        # Current estimates
        _, values = self.model(states)
        _, next_values = self.model(next_states)

        values = values.numpy().flatten()
        next_values = next_values.numpy().flatten()

        # Compute advantages
        advantages = self.compute_advantages(rewards, values, next_values, dones)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # One training step
        with tf.GradientTape() as tape:
            new_policies, new_values = self.model(states)
            new_values = tf.squeeze(new_values)

            # Select the probability of the chosen actions
            action_masks = tf.one_hot(actions, self.action_dim)
            new_probs = tf.reduce_sum(new_policies * action_masks, axis=1)
            old_probs = tf.reduce_sum(old_policies * action_masks, axis=1)

            # Probability ratio
            ratio = tf.exp(tf.math.log(new_probs + 1e-10) - tf.math.log(old_probs + 1e-10))

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            critic_loss = tf.reduce_mean((returns - new_values) ** 2)

            entropy = tf.reduce_mean(-new_policies * tf.math.log(new_policies + 1e-10))
            total_loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return total_loss.numpy()