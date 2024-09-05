import unittest
import numpy as np
import os
from board_game_env import BoardGameEnv
from q_learning_agent import QLearningAgent

class TestBoardGameEnv(unittest.TestCase):
    def setUp(self):
        print("Setting up TestBoardGameEnv")
        self.env = BoardGameEnv()

    def test_init(self):
        print("Running test_init")
        self.assertEqual(self.env.board_size, 8)
        self.assertEqual(self.env.board.shape, (8, 8))
        self.assertEqual(self.env.current_player, 1)

    def test_reset(self):
        print("Running test_reset")
        self.env.board = np.ones((8, 8))
        self.env.current_player = -1
        state = self.env.reset()
        np.testing.assert_array_equal(state, np.zeros((8, 8)))
        self.assertEqual(self.env.current_player, 1)

    def test_step(self):
        print("Running test_step")
        action = 0
        next_state, reward, done, _ = self.env.step(action)
        self.assertEqual(next_state[0, 0], 1)
        self.assertEqual(self.env.current_player, -1)

    def test_check_win(self):
        print("Running test_check_win")
        self.env.board[0, :] = 1
        self.assertTrue(self.env.check_win())

    def test_get_valid_actions(self):
        print("Running test_get_valid_actions")
        self.env.board[0, 0] = 1
        valid_actions = self.env.get_valid_actions()
        self.assertNotIn(0, valid_actions)
        self.assertEqual(len(valid_actions), 63)

class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        print("Setting up TestQLearningAgent")
        self.agent = QLearningAgent(state_size=64, action_size=64)

    def test_init(self):
        print("Running test_init")
        self.assertEqual(self.agent.state_size, 64)
        self.assertEqual(self.agent.action_size, 64)
        self.assertEqual(len(self.agent.q_table), 0)

    def test_get_q_value(self):
        print("Running test_get_q_value")
        state = np.zeros(64)
        action = 0
        self.assertEqual(self.agent.get_q_value(state, action), 0.0)

    def test_update_q_value(self):
        print("Running test_update_q_value")
        state = np.zeros(64)
        next_state = np.zeros(64)
        next_state[0] = 1
        action = 0
        reward = 1
        self.agent.update_q_value(state, action, reward, next_state)
        self.assertGreater(self.agent.get_q_value(state, action), 0.0)

    def test_choose_action(self):
        print("Running test_choose_action")
        state = np.zeros(64)
        valid_actions = list(range(64))
        action = self.agent.choose_action(state, valid_actions)
        self.assertIn(action, valid_actions)

    def test_save_load_model(self):
        print("Running test_save_load_model")
        self.agent.q_table[(0, 0)] = 1.0
        self.agent.save_model("test_model.json")
        self.assertTrue(os.path.exists("test_model.json"))

        new_agent = QLearningAgent(state_size=64, action_size=64)
        new_agent.load_model("test_model.json")
        self.assertEqual(new_agent.q_table[(0, 0)], 1.0)

        os.remove("test_model.json")

if __name__ == "__main__":
    unittest.main()