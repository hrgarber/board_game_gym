import unittest

import numpy as np

from src.environments.board_game_env import BoardGameEnv


class TestCase(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = (
            self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        )
        self.action_size = self.env.action_space.n

    def assert_valid_action(self, action):
        self.assertIsInstance(action, (int, np.int64))
        self.assertTrue(0 <= action < self.action_size)

    def assert_valid_state(self, state):
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (self.state_size,))

    def assert_valid_reward(self, reward):
        self.assertIsInstance(reward, (int, float, np.float32, np.float64))

    def assert_valid_done(self, done):
        self.assertIsInstance(done, (bool, np.bool_))


if __name__ == "__main__":
    unittest.main()
