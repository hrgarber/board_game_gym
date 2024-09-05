from tests.test_utils import TestCase
import numpy as np
import os

class TestBoardGameEnv(TestCase):
    def test_init(self):
        self.assertEqual(self.env.board_size, 8)
        self.assertEqual(self.env.board.shape, (8, 8))
        self.assertEqual(self.env.current_player, 1)

    def test_reset(self):
        self.env.board = np.ones((8, 8))
        self.env.current_player = -1
        state = self.env.reset()
        self.assert_valid_state(state)
        np.testing.assert_array_equal(state.reshape(8, 8), np.zeros((8, 8)))
        self.assertEqual(self.env.current_player, 1)

    def test_step(self):
        initial_state = self.env.reset()
        action = 0
        next_state, reward, done, _ = self.env.step(action)
        self.assert_valid_state(next_state)
        self.assert_valid_reward(reward)
        self.assert_valid_done(done)
        self.assertEqual(next_state.reshape(8, 8)[0, 0], 1)
        self.assertEqual(self.env.current_player, -1)

    def test_check_win(self):
        self.env.board[0, :5] = 1
        self.assertTrue(self.env.check_win(0, 4))
        self.env.board = np.zeros((self.env.board_size, self.env.board_size))
        self.assertFalse(self.env.check_win(0, 0))

    def test_get_valid_actions(self):
        self.env.board[0, 0] = 1
        valid_actions = self.env.get_valid_actions()
        self.assertNotIn(0, valid_actions)
        self.assertEqual(len(valid_actions), 63)

class TestQLearningAgent(TestCase):
    def setUp(self):
        super().setUp()
        self.agent = self.create_q_learning_agent()

    def test_init(self):
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
        self.assert_valid_action(action)

    def test_save_load_model(self):
        self.agent.q_table[(0, 0)] = 1.0
        self.agent.save_model("test_model.json")
        self.assertTrue(os.path.exists("test_model.json"))

        new_agent = self.create_q_learning_agent()
        new_agent.load_model("test_model.json")
        self.assertEqual(new_agent.q_table[(0, 0)], 1.0)

        os.remove("test_model.json")

if __name__ == "__main__":
    from unittest import main
    main()
