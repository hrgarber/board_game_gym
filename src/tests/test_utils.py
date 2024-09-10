import unittest
import numpy as np
from src.backend.environments.board_game_env import BoardGameEnv
from src.backend.logger import logger

class TestCase(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up test environment")
        self.env = BoardGameEnv()
        self.state_size = (
            self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        )
        self.action_size = self.env.action_space.n
        logger.debug(f"State size: {self.state_size}, Action size: {self.action_size}")

    def tearDown(self):
        logger.info("Tearing down test environment")

    def assert_valid_action(self, action):
        logger.info(f"Asserting valid action: {action}")
        self.assertIsInstance(action, (int, np.int64))
        self.assertTrue(0 <= action < self.action_size)
        logger.debug(f"Action {action} is valid")

    def assert_valid_state(self, state):
        logger.info("Asserting valid state")
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (self.state_size,))
        logger.debug(f"State shape {state.shape} is valid")

    def assert_valid_reward(self, reward):
        logger.info(f"Asserting valid reward: {reward}")
        self.assertIsInstance(reward, (int, float, np.float32, np.float64))
        logger.debug(f"Reward {reward} is valid")

    def assert_valid_done(self, done):
        logger.info(f"Asserting valid done flag: {done}")
        self.assertIsInstance(done, (bool, np.bool_))
        logger.debug(f"Done flag {done} is valid")

    def test_env_initialization(self):
        logger.info("Testing environment initialization")
        self.assert_valid_state(self.env.reset())
        logger.debug("Environment initialized successfully")

    def test_env_step(self):
        logger.info("Testing environment step")
        initial_state = self.env.reset()
        action = self.env.action_space.sample()
        logger.debug(f"Sampled action: {action}")
        next_state, reward, done, _ = self.env.step(action)
        self.assert_valid_state(next_state)
        self.assert_valid_reward(reward)
        self.assert_valid_done(done)
        logger.debug("Environment step executed successfully")

    def test_valid_actions(self):
        logger.info("Testing valid actions")
        valid_actions = self.env.get_valid_actions()
        logger.debug(f"Valid actions: {valid_actions}")
        for action in valid_actions:
            self.assert_valid_action(action)
        logger.debug("All actions are valid")

if __name__ == "__main__":
    unittest.main()
