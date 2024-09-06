import unittest
import torch
import numpy as np
import os
from src.agents.dqn_agent import DQNAgent
from src.environments.board_game_env import BoardGameEnv

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.action_size = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DQNAgent(self.state_size, self.action_size, self.device)

    def test_initialization(self):
        self.assertIsInstance(self.agent.model, torch.nn.Module)
        self.assertIsInstance(self.agent.target_model, torch.nn.Module)
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)

    def test_preprocess_state(self):
        state = np.random.rand(self.state_size)
        processed_state = self.agent.preprocess_state(state)
        self.assertIsInstance(processed_state, torch.FloatTensor)
        self.assertEqual(processed_state.device, self.device)

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
        self.agent.replay(self.agent.batch_size)
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

    def test_save_load_model(self):
        # Save the model
        self.agent.save("test_dqn_model.pth")
        self.assertTrue(os.path.exists("test_dqn_model.pth"))

        # Create a new agent and load the saved model
        new_agent = DQNAgent(self.state_size, self.action_size, self.device)
        new_agent.load("test_dqn_model.pth")

        # Check if the loaded model has the same weights as the original model
        for param1, param2 in zip(self.agent.model.parameters(), new_agent.model.parameters()):
            self.assertTrue(torch.equal(param1, param2))

        # Clean up
        os.remove("test_dqn_model.pth")

    def test_update(self):
        state = np.random.rand(self.state_size)
        action = np.random.randint(self.action_size)
        reward = np.random.rand()
        next_state = np.random.rand(self.state_size)
        done = False

        initial_memory_length = len(self.agent.memory)
        self.agent.update(state, action, reward, next_state, done)
        self.assertEqual(len(self.agent.memory), initial_memory_length + 1)

    def test_train(self):
        num_episodes = 5
        max_steps = 10
        self.agent.train(self.env, num_episodes, max_steps)
        self.assertGreater(len(self.agent.memory), 0)

    def test_epsilon_decay(self):
        initial_epsilon = self.agent.epsilon
        for _ in range(100):
            self.agent.replay(self.agent.batch_size)
            self.agent.decay_epsilon()  # Explicitly call decay_epsilon after each replay
        self.assertLess(self.agent.epsilon, initial_epsilon)

    def test_model_output(self):
        state = torch.randn(1, self.state_size).to(self.device)
        output = self.agent.model(state)
        self.assertEqual(output.shape, (1, self.action_size))

    def test_target_model_update_frequency(self):
        initial_target_weights = self.agent.target_model.fc1.weight.data.clone()
        for _ in range(self.agent.update_target_every):
            self.agent.train(self.env, 1, 1)
        self.assertFalse(torch.equal(initial_target_weights, self.agent.target_model.fc1.weight.data))

if __name__ == '__main__':
    unittest.main()

