import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    LEARNING_RATE,
    DISCOUNT_FACTOR,
    EPSILON,
    EPSILON_MIN,
    EPSILON_DECAY,
    BATCH_SIZE,
    UPDATE_TARGET_EVERY,
    DEVICE,
)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = x.view(-1, self.state_size)  # Ensure correct input shape
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        update_target_every=UPDATE_TARGET_EVERY,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = deque(maxlen=10000)
        self.update_target_every = update_target_every
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        self.version = 1

    def preprocess_state(self, state):
        return torch.FloatTensor(state).to(self.device)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self.preprocess_state(state)
        act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0  # Return 0 if no replay was performed

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.stack([self.preprocess_state(state) for state in states])
        next_states = torch.stack(
            [self.preprocess_state(state) for state in next_states]
        )

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Double DQN: use online network to select action, target network to evaluate it
        next_actions = self.model(next_states).max(1)[1].unsqueeze(1)
        next_q_values = (
            self.target_model(next_states).gather(1, next_actions).squeeze(1).detach()
        )
        target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()  # Return the loss value for monitoring

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, name):
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.version = checkpoint["version"]

    def save(self, name):
        torch.save(
            {"model_state_dict": self.model.state_dict(), "version": self.version}, name
        )
        self.version += 1

    def update(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)

    def train(self, env, num_episodes, max_steps):
        for episode in range(num_episodes):
            state = env.reset()
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

            self.decay_epsilon()

            if (episode + 1) % self.update_target_every == 0:
                self.update_target_model()

        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)
