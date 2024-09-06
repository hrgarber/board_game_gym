from tests.test_utils import TestCase
import numpy as np
import math
from game_files.game_bot import GameBot
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

    def test_is_draw(self):
        self.env.board = np.ones((self.env.board_size, self.env.board_size))
        self.assertTrue(self.env.is_draw())
        self.env.board[0, 0] = 0
        self.assertFalse(self.env.is_draw())

    def test_check_blocking_move(self):
        self.env.board[0, :4] = 1
        self.assertTrue(self.env.check_blocking_move(0, 4))
        self.assertFalse(self.env.check_blocking_move(1, 0))
        
        # Additional test for the case (0, 4)
        self.env.board = np.zeros((self.env.board_size, self.env.board_size))
        self.env.board[0, :4] = -1  # Opponent's pieces
        self.assertTrue(self.env.check_blocking_move(0, 4))
        
        # Test blocking a potential win
        self.env.board = np.zeros((self.env.board_size, self.env.board_size))
        self.env.board[0, :4] = 1
        self.assertTrue(self.env.check_blocking_move(0, 4))

    def test_check_line(self):
        self.env.board[0, :4] = 1
        self.assertTrue(self.env.check_line(0, 3, 4))
        self.assertFalse(self.env.check_line(0, 3, 5))

    def test_render(self):
        self.env.reset()
        self.env.board[0, 0] = 1
        self.env.board[1, 1] = -1
        rendered_output = self.env.render(mode='ansi')
        self.assertIsInstance(rendered_output, str)
        self.assertIn('X', rendered_output)
        self.assertIn('O', rendered_output)

class TestAlphaBetaPruning(TestCase):
    def setUp(self):
        super().setUp()
        self.game_bot = GameBot(board_size=3)

    def test_alpha_beta_pruning(self):
        # Set up a simple board state
        self.game_bot.board = [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ]
        self.game_bot.current_player = 1

        # Test the alpha_beta_pruning method
        best_score = self.game_bot.alpha_beta_pruning(depth=3, alpha=-math.inf, beta=math.inf, maximizing_player=True)
        self.assertIsInstance(best_score, (int, float))

    def test_get_best_move(self):
        # Set up a simple board state
        self.game_bot.board = [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 0]
        ]
        self.game_bot.current_player = 1

        # Test the get_best_move method
        best_move = self.game_bot.get_best_move(depth=3)
        self.assertIsInstance(best_move, tuple)
        self.assertEqual(len(best_move), 2)
        row, col = best_move
        self.assertTrue(0 <= row < 3)
        self.assertTrue(0 <= col < 3)

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
