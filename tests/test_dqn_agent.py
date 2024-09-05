from tests.test_utils import TestCase
import torch
import numpy as np
import os

class TestDQNAgent(TestCase):
    def setUp(self):
        super().setUp()
        self.agent = self.create_dqn_agent()

    def test_dqn_initialization(self):
        self.assertIsInstance(self.agent.model, torch.nn.Module)
        self.assertIsInstance(self.agent.target_model, torch.nn.Module)
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
        self.assert_valid_action(action)

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
        new_agent = self.create_dqn_agent()
        new_agent.load("test_dqn_model.pth")

        # Check if the loaded model has the same weights as the original model
        for param1, param2 in zip(self.agent.model.parameters(), new_agent.model.parameters()):
            self.assertTrue(torch.equal(param1, param2))

        # Clean up
        os.remove("test_dqn_model.pth")

if __name__ == '__main__':
    from unittest import main
    main()
