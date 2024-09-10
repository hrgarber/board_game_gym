import unittest
import numpy as np
from src.backend.environments.board_game_env import BoardGameEnv
from src.backend.game import BoardGame
from src.backend.logger import logger

class TestBoardGame(unittest.TestCase):
    def setUp(self):
        self.game = BoardGame()
        logger.info("Setting up test environment")

    def tearDown(self):
        logger.info("Tearing down test environment")

    def test_board_size(self):
        logger.info("Testing board size")
        self.assertEqual(self.game.rows, 8, "Board height should be 8")
        self.assertEqual(self.game.cols, 12, "Board width should be 12")
        self.assertEqual(self.game.board.shape, (8, 12), "Board shape should be (8, 12)")

    def test_regular_piece_placement(self):
        logger.info("Testing regular piece placement")
        self.assertTrue(self.game.make_move(0, 0, False), "Regular piece should be placed successfully")
        self.assertEqual(self.game.board[0, 0], 1, "Regular piece should be placed with value 1")

    def test_permanent_piece_placement(self):
        logger.info("Testing permanent piece placement")
        self.assertTrue(self.game.make_move(0, 0, True), "Permanent piece should be placed successfully")
        self.assertEqual(self.game.board[0, 0], -1, "Permanent piece should be placed with opponent's color")
        self.assertTrue(self.game.permanent_pieces[0, 0], "Permanent piece should be marked as permanent")

    def test_flipping_mechanic(self):
        logger.info("Testing flipping mechanic")
        self.game.board[0, 0] = 1  # Place regular piece
        self.game.board[0, 1] = -1  # Place opponent's regular piece
        self.game.make_move(1, 1, True)  # Place permanent piece
        self.assertEqual(self.game.board[0, 0], -1, "Adjacent regular piece should be flipped")
        self.assertEqual(self.game.board[0, 1], 1, "Adjacent regular piece should be flipped")
        self.assertEqual(self.game.board[1, 1], -1, "Permanent piece should be placed with opponent's color")

    def test_win_condition(self):
        logger.info("Testing win condition")
        # Create a path from left to right
        for i in range(12):
            self.game.board[0, i] = 1
        self.assertEqual(self.game.check_winner(), 1, "Player 1 should win with a complete path")

    def test_no_win_condition(self):
        logger.info("Testing no win condition")
        # Create an incomplete path
        for i in range(11):
            self.game.board[0, i] = 1
        self.assertEqual(self.game.check_winner(), 0, "No player should win with an incomplete path")

    def test_diagonal_path(self):
        logger.info("Testing diagonal path")
        for i in range(8):
            self.game.board[i, i] = 1
        self.game.board[7, 11] = 1
        self.assertEqual(self.game.check_winner(), 1, "Player 1 should win with a diagonal path")

    def test_zigzag_path(self):
        logger.info("Testing zigzag path")
        path = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (6, 0), (7, 1), (7, 11)]
        for row, col in path:
            self.game.board[row, col] = 1
        self.assertEqual(self.game.check_winner(), 1, "Player 1 should win with a zigzag path")

class TestBoardGameEnv(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        logger.info("Setting up test environment")

    def tearDown(self):
        logger.info("Tearing down test environment")

    def test_reset(self):
        logger.info("Testing reset")
        self.env.game.board[0, 0] = 1
        state = self.env.reset()
        self.assertTrue(np.all(state == 0), "Board should be reset to all zeros")

    def test_step_regular_piece(self):
        logger.info("Testing step with regular piece")
        action = 0  # Place regular piece at (0, 0)
        state, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.game.board[0, 0], 1, "Regular piece should be placed with value 1")
        self.assertEqual(reward, 0, "Reward should be 0 for a non-winning move")
        self.assertFalse(done, "Game should not be over")

    def test_step_permanent_piece(self):
        logger.info("Testing step with permanent piece")
        action = 1  # Place permanent piece at (0, 0)
        state, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.game.board[0, 0], -1, "Permanent piece should be placed with opponent's color")
        self.assertTrue(self.env.game.permanent_pieces[0, 0], "Permanent piece should be marked as permanent")

    def test_invalid_move(self):
        logger.info("Testing invalid move")
        self.env.step(0)  # Place a piece at (0, 0)
        state, reward, done, _ = self.env.step(0)  # Try to place another piece at (0, 0)
        self.assertEqual(reward, -10, "Reward should be -10 for an invalid move")
        self.assertTrue(done, "Game should be over after an invalid move")

    def test_win_condition(self):
        logger.info("Testing win condition")
        # Create a winning path
        for i in range(11):
            self.env.step(i * 2)  # Place regular pieces
        state, reward, done, _ = self.env.step(22)  # Complete the path
        self.assertEqual(reward, 10, "Reward should be 10 for a winning move")
        self.assertTrue(done, "Game should be over after a win")

    def test_get_valid_actions(self):
        logger.info("Testing get_valid_actions")
        valid_actions = self.env.get_valid_actions()
        self.assertEqual(len(valid_actions), self.env.game.rows * self.env.game.cols * 2, "All actions should be valid on an empty board")

        self.env.step(0)  # Place a regular piece
        valid_actions = self.env.get_valid_actions()
        self.assertEqual(len(valid_actions), self.env.game.rows * self.env.game.cols * 2 - 1, "One less action should be available after a regular move")

if __name__ == "__main__":
    unittest.main()

def create_suite():
    logger.info("Creating test suite for BoardGame and BoardGameEnv")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBoardGame))
    suite.addTest(unittest.makeSuite(TestBoardGameEnv))
    return suite
