import json

import numpy as np

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    EPSILON,
    EPSILON_DECAY,
    EPSILON_MIN,
)


class QLearningAgent:
    """
    A Q-Learning agent for reinforcement learning.
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.version = 1
        self.training_episodes = 0

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_value(self, state, action):
        state_key = self._get_state_key(state)
        return self.q_table.get((state_key, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)

        current_q = self.get_q_value(state, action)
        next_max_q = max(
            [self.get_q_value(next_state, a) for a in range(self.action_size)]
        )

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[(state_key, action)] = new_q

    def choose_action(self, state, valid_actions):
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))
        else:
            q_values = [self.get_q_value(state, action) for action in valid_actions]
            return int(valid_actions[np.argmax(q_values)])

    def act(self, state):
        valid_actions = list(range(self.action_size))
        return self.choose_action(state, valid_actions)

    def _get_state_key(self, state):
        return tuple(state.flatten())

    def save_model(self, filename):
        model_data = {
            "q_table": {str(k): v for k, v in self.q_table.items()},
            "version": self.version,
            "training_episodes": self.training_episodes,
            "epsilon": self.epsilon,
        }
        with open(filename, "w") as f:
            json.dump(model_data, f)
        self.version += 1

    def load_model(self, filename):
        with open(filename, "r") as f:
            model_data = json.load(f)
        self.q_table = {eval(k): v for k, v in model_data["q_table"].items()}
        self.version = model_data["version"]
        self.training_episodes = model_data["training_episodes"]
        self.epsilon = model_data["epsilon"]

    def update(self, state, action, reward, next_state, done):
        self.update_q_value(state, action, reward, next_state)

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_action(state, env.get_valid_actions())
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
            self.decay_epsilon()
            self.training_episodes += 1
