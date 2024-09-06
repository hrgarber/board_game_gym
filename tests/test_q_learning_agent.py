import unittest
import numpy as np
from src.agents.q_learning_agent import QLearningAgent
from src.environments.board_game_env import BoardGameEnv

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.action_size = self.env.action_space.n
        self.agent = QLearningAgent(self.state_size, self.action_size)

    def test_initialization(self):
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(len(self.agent.q_table), 0)

    def test_get_q_value(self):
        state = np.zeros(self.state_size)
        action = 0
        self.assertEqual(self.agent.get_q_value(state, action), 0.0)

    def test_update_q_value(self):
        state = np.zeros(self.state_size)
        next_state = np.zeros(self.state_size)
        next_state[0] = 1
        action = 0
        reward = 1
        self.agent.update_q_value(state, action, reward, next_state)
        self.assertGreater(self.agent.get_q_value(state, action), 0.0)

    def test_choose_action(self):
        state = np.zeros(self.state_size)
        valid_actions = list(range(self.action_size))
        action = self.agent.choose_action(state, valid_actions)
        self.assertIn(action, valid_actions)

    def test_act(self):
        state = np.zeros(self.state_size)
        action = self.agent.act(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)

    def test_decay_epsilon(self):
        initial_epsilon = self.agent.epsilon
        self.agent.decay_epsilon()
        self.assertLess(self.agent.epsilon, initial_epsilon)

    def test_update(self):
        state = np.zeros(self.state_size)
        next_state = np.zeros(self.state_size)
        next_state[0] = 1
        action = 0
        reward = 1
        done = False
        self.agent.update(state, action, reward, next_state, done)
        self.assertGreater(self.agent.get_q_value(state, action), 0.0)

    def test_train(self):
        num_episodes = 10
        self.agent.train(self.env, num_episodes)
        self.assertEqual(self.agent.training_episodes, num_episodes)
        self.assertGreater(len(self.agent.q_table), 0)

if __name__ == '__main__':
    unittest.main()
