import numpy as np
import json
import os
from datetime import datetime
import logging
import platform
import torch
from ..environments.board_game_env import BoardGameEnv

class QLearningAgent:
    """
    Q-Learning agent for playing the board game.
    This agent learns to make decisions based on the current state of the game board.
    """

    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize the Q-Learning agent.

        Args:
            state_size (int): The size of the state space (board_size^2).
            action_size (int): The size of the action space (board_size^2).
            learning_rate (float): The learning rate for Q-value updates.
            discount_factor (float): The discount factor for future rewards.
            epsilon (float): The exploration rate for the epsilon-greedy policy.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        self.device = torch.device("cuda" if torch.cuda.is_available() and platform.system() == "Windows" else "cpu")
        self.q_table = torch.zeros((state_size, action_size), device=self.device)
        self.training_history = []
        self.version = "1.0.0"
        self.total_training_episodes = 0

    def get_q_value(self, state, action):
        """
        Get the Q-value for a given state-action pair.

        Args:
            state (numpy.array): The current state of the game board.
            action (int): The action to evaluate.

        Returns:
            float: The Q-value for the given state-action pair.
        """
        state_key = self.state_to_key(state)
        return self.q_table[state_key, action].item()

    def update_q_value(self, state, action, reward, next_state):
        """
        Update the Q-value for a given state-action pair.

        Args:
            state (numpy.array): The current state of the game board.
            action (int): The action taken.
            reward (float): The reward received for taking the action.
            next_state (numpy.array): The resulting state after taking the action.
        """
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)
        
        old_q = self.q_table[state_key, action]
        next_max_q = torch.max(self.q_table[next_state_key])
        
        new_q = old_q + self.learning_rate * (reward + self.discount_factor * next_max_q - old_q)
        self.q_table[state_key, action] = new_q

    def choose_action(self, state, valid_actions):
        """
        Choose an action using an epsilon-greedy policy.

        Args:
            state (numpy.array): The current state of the game board.
            valid_actions (list): A list of valid actions to choose from.

        Returns:
            int: The chosen action.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            state_key = self.state_to_key(state)
            q_values = self.q_table[state_key, valid_actions]
            max_q = torch.max(q_values)
            actions_with_max_q = [action for action, q_value in zip(valid_actions, q_values) if q_value == max_q]
            return np.random.choice(actions_with_max_q)

    def state_to_key(self, state):
        """
        Convert a state (game board) to a hashable key for the Q-table.

        Args:
            state (numpy.array): The game board state to convert.

        Returns:
            int: A hashable representation of the state.
        """
        return np.ravel_multi_index(state.flatten().astype(int), (3,) * self.state_size)

    def save_model(self, filename):
        """
        Save the Q-table and training history to a file.

        Args:
            filename (str): The name of the file to save the model to.
        """
        model_data = {
            "q_table": self.q_table.cpu().numpy().tolist(),
            "training_history": self.training_history,
            "version": self.version,
            "total_training_episodes": self.total_training_episodes,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filename):
        """
        Load the Q-table and training history from a file.

        Args:
            filename (str): The name of the file to load the model from.
        """
        try:
            with open(filename, 'r') as f:
                model_data = json.load(f)
            
            self.q_table = torch.tensor(model_data["q_table"], device=self.device)
            self.training_history = model_data["training_history"]
            self.version = model_data["version"]
            self.total_training_episodes = model_data["total_training_episodes"]
            self.learning_rate = model_data["learning_rate"]
            self.discount_factor = model_data["discount_factor"]
            self.epsilon = model_data["epsilon"]
        except json.JSONDecodeError as e:
            logging.error(f"JSON Decode Error in file {filename}: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error loading model from {filename}: {str(e)}")
            raise

    def update_training_history(self, episode, win_rate):
        """
        Update the training history with the latest episode results.

        Args:
            episode (int): The current episode number.
            win_rate (float): The win rate for the last batch of episodes.
        """
        self.training_history.append({
            "episode": episode,
            "win_rate": win_rate,
            "timestamp": datetime.now().isoformat()
        })
        self.total_training_episodes = episode

    def get_latest_version(self, directory):
        """
        Get the latest version number from saved models in the specified directory.

        Args:
            directory (str): The directory to search for saved models.

        Returns:
            str: The latest version number.
        """
        versions = []
        for filename in os.listdir(directory):
            if filename.startswith("model_v") and filename.endswith(".json"):
                version = filename.split("_v")[1].split(".json")[0]
                versions.append(version)
        
        if versions:
            return max(versions)
        else:
            return "1.0.0"

    def save_model_with_version(self, directory):
        """
        Save the model with an incremented version number.

        Args:
            directory (str): The directory to save the model in.

        Returns:
            str: The filename of the saved model.
        """
        latest_version = self.get_latest_version(directory)
        major, minor, patch = map(int, latest_version.split('.'))
        new_version = f"{major}.{minor}.{patch + 1}"
        self.version = new_version
        
        filename = os.path.join(directory, f"model_v{new_version}.json")
        self.save_model(filename)
        return filename

    def load_latest_model(self, directory):
        """
        Load the latest version of the model from the specified directory.

        Args:
            directory (str): The directory to search for saved models.

        Returns:
            bool: True if a model was loaded, False otherwise.
        """
        latest_version = self.get_latest_version(directory)
        filename = os.path.join(directory, f"model_v{latest_version}.json")
        
        if os.path.exists(filename):
            try:
                self.load_model(filename)
                return True
            except json.JSONDecodeError:
                logging.error(f"JSON Decode Error when loading {filename}. File may be corrupted.")
                return False
            except Exception as e:
                logging.error(f"Error loading model from {filename}: {str(e)}")
                return False
        else:
            return False
