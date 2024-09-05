import unittest
import torch
import numpy as np
from src.agents.dqn_agent import DQNAgent, DQN

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.state_size = 9
        self.action_size = 9
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DQNAgent(self.state_size, self.action_size, self.device)

    def test_dqn_initialization(self):
        self.assertIsInstance(self.agent.model, DQN)
        self.assertIsInstance(self.agent.target_model, DQN)
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)

    def test_remember(self):
        initial_memory_length = len(self.agent.memory)
        state = np.random.rand(self.state_size)
        action = np.random.randint(self.action_size)
        reward = np.random.rand()
        next_state = np.random.rand(self.state_size)
        done = False

        self.agent.remember(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_memory_length + 1)

    def test_act(self):
        state = np.random.rand(self.state_size)
        action = self.agent.act(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)

    def test_replay(self):
        # Fill the memory with some sample experiences
        for _ in range(self.agent.batch_size + 1):
            state = np.random.rand(self.state_size)
            action = np.random.randint(self.action_size)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_size)
            done = False
            self.agent.remember(state, action, reward, next_state, done)

        # Perform a replay step
        initial_q_values = self.agent.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()
        self.agent.replay()
        updated_q_values = self.agent.model(torch.FloatTensor(state).unsqueeze(0).to(self.device)).detach().cpu().numpy()

        # Check if Q-values have been updated
        self.assertFalse(np.array_equal(initial_q_values, updated_q_values))

    def test_update_target_model(self):
        # Get initial weights
        initial_weights = self.agent.target_model.fc1.weight.data.clone()

        # Update the model weights
        self.agent.model.fc1.weight.data += 1.0

        # Update target model
        self.agent.update_target_model()

        # Check if target model weights have been updated
        updated_weights = self.agent.target_model.fc1.weight.data
        self.assertFalse(torch.equal(initial_weights, updated_weights))

if __name__ == '__main__':
    unittest.main()
