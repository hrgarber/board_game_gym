import numpy as np
import torch
import json

class QLearningAgent:
    """
    A Q-Learning agent for reinforcement learning.
    """

    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize the Q-Learning agent.

        Args:
            state_size (int): The size of the state space.
            action_size (int): The number of possible actions.
            learning_rate (float): The learning rate for Q-value updates.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The exploration rate for epsilon-greedy policy.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.version = 1

    def get_q_value(self, state, action):
        """
        Get the Q-value for a given state-action pair.

        Args:
            state (numpy.array): The current state.
            action (int): The action to evaluate.

        Returns:
            float: The Q-value for the given state-action pair.
        """
        state_key = self._get_state_key(state)
        return self.q_table.get((state_key, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a given state-action pair.

        Args:
            state (numpy.array): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (numpy.array): The resulting state after taking the action.
        """
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        current_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        self.q_table[(state_key, action)] = new_q

    def choose_action(self, state, valid_actions):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (numpy.array): The current state.
            valid_actions (list): List of valid actions.

        Returns:
            int: The chosen action.
        """
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))
        else:
            q_values = [self.get_q_value(state, action) for action in valid_actions]
            return int(valid_actions[np.argmax(q_values)])

    def act(self, state):
        """
        Choose an action for the given state.

        Args:
            state (numpy.array): The current state.

        Returns:
            int: The chosen action.
        """
        valid_actions = list(range(self.action_size))
        return self.choose_action(state, valid_actions)

    def _get_state_key(self, state):
        """
        Convert a state array to a hashable key for the Q-table.

        Args:
            state (numpy.array): The state to convert.

        Returns:
            tuple: A hashable representation of the state.
        """
        return tuple(state.flatten())

    def save_model(self, filename):
        """
        Save the Q-learning model to a file.

        Args:
            filename (str): The name of the file to save the model to.
        """
        model_data = {
            "q_table": {str(k): v for k, v in self.q_table.items()},
            "version": self.version
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filename):
        """
        Load a Q-learning model from a file.

        Args:
            filename (str): The name of the file to load the model from.
        """
        with open(filename, 'r') as f:
            model_data = json.load(f)
        self.q_table = {eval(k): v for k, v in model_data["q_table"].items()}
        self.version = model_data["version"]
