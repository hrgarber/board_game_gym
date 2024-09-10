import unittest
import numpy as np
from src.backend.environments.board_game_env import BoardGameEnv
from src.backend.logger import logger

class TestCase(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up test environment")
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0]
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
        self.assertEqual(len(valid_actions), self.env.game.rows * self.env.game.cols * 2)
        logger.debug("All actions are valid")

    def test_board_dimensions(self):
        logger.info("Testing board dimensions")
        self.assertEqual(self.env.game.cols, 12, "Board width should be 12")
        self.assertEqual(self.env.game.rows, 8, "Board height should be 8")
        self.assertEqual(self.state_size, 8 * 12, "State size should be 8 * 12")
        logger.debug("Board dimensions are correct")

    def test_action_space(self):
        logger.info("Testing action space")
        self.assertEqual(self.action_size, self.env.game.rows * self.env.game.cols * 2, 
                         "Action space should be twice the number of board cells")
        logger.debug("Action space is correct")

    def test_observation_space(self):
        logger.info("Testing observation space")
        self.assertEqual(self.env.observation_space.shape, (8 * 12,), 
                         "Observation space shape should be (96,)")
        self.assertTrue(np.all(self.env.observation_space.low == -1), "Lowest observation value should be -1")
        self.assertTrue(np.all(self.env.observation_space.high == 1), "Highest observation value should be 1")
        logger.debug("Observation space is correct")

    def test_reset(self):
        logger.info("Testing reset function")
        self.env.step(0)  # Make a move
        reset_state = self.env.reset()
        self.assert_valid_state(reset_state)
        self.assertTrue(np.all(self.env.game.board == 0), "Board should be empty after reset")
        self.assertEqual(self.env.game.current_player, 1, "Current player should be 1 after reset")
        logger.debug("Reset function works correctly")

if __name__ == "__main__":
    unittest.main()
