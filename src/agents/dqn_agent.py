import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from src.environments.board_game_env import BoardGameEnv


class DQN(nn.Module):
    """Deep Q-Network model."""

    def __init__(self, state_size, action_size):
        """Initialize the DQN model.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """DQN Agent for training and decision making."""

    def __init__(self, state_size, action_size, device):
        """Initialize the DQN Agent.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Dimension of the action space.
            device (torch.device): Device to run the model on (CPU or CUDA).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory.

        Args:
            state (numpy.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy.array): Next state.
            done (bool): Whether the episode is done.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using an epsilon-greedy policy.

        Args:
            state (numpy.array): Current state.

        Returns:
            int: Chosen action.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state_tensor)
        return action_values.argmax().item()

    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """Update the target model with the trained model's weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, filename):
        """Save the model's state dict.

        Args:
            filename (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Load the model's state dict.

        Args:
            filename (str): Path to load the model from.
        """
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(self.model.state_dict())
