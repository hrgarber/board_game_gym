import unittest
import numpy as np
from src.backend.environments.board_game_env import BoardGameEnv
from src.backend.logger import logger

class TestBoardGameEnv(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        logger.info("Setting up test environment")

    def tearDown(self):
        logger.info("Tearing down test environment")

    def test_board_size(self):
        logger.info("Testing board size")
        self.assertEqual(self.env.board_width, 12, "Board width should be 12")
        self.assertEqual(self.env.board_height, 8, "Board height should be 8")
        self.assertEqual(self.env.board.shape, (8, 12), "Board shape should be (8, 12)")

    def test_regular_piece_placement(self):
        logger.info("Testing regular piece placement")
        action = 0  # Place regular piece at (0, 0)
        _, _, _, _ = self.env.step(action)
        self.assertEqual(self.env.board[0, 0], 1, "Regular piece should be placed with value 1")

    def test_permanent_piece_placement(self):
        logger.info("Testing permanent piece placement")
        action = 1  # Place permanent piece at (0, 0)
        _, _, _, _ = self.env.step(action)
        self.assertEqual(self.env.board[0, 0], -2, "Permanent piece should be placed with value -2")

    def test_flipping_mechanic(self):
        logger.info("Testing flipping mechanic")
        self.env.board[0, 0] = 1  # Place regular piece
        self.env.board[0, 1] = -1  # Place opponent's regular piece
        action = 3  # Place permanent piece at (0, 1)
        _, _, _, _ = self.env.step(action)
        self.assertEqual(self.env.board[0, 0], -1, "Adjacent regular piece should be flipped")
        self.assertEqual(self.env.board[0, 1], 2, "Permanent piece should not be flipped")

    def test_win_condition(self):
        logger.info("Testing win condition")
        # Create a path from top to bottom
        for i in range(8):
            self.env.board[i, 0] = 1
        self.assertTrue(self.env.check_win(), "Player should win with a complete path")

    def test_no_win_condition(self):
        logger.info("Testing no win condition")
        # Create an incomplete path
        for i in range(7):
            self.env.board[i, 0] = 1
        self.assertFalse(self.env.check_win(), "Player should not win with an incomplete path")

    def test_edge_rule(self):
        logger.info("Testing edge rule")
        action = 1  # Place permanent piece at (0, 0) (side A)
        _, _, _, _ = self.env.step(action)
        self.assertEqual(self.env.board[0, 0], -2, "Permanent piece on side A should be opponent's color")

        self.env.current_player = -1  # Switch to the other player
        action = (self.env.board_height - 1) * self.env.board_width * 2 + 1  # Place permanent piece at (7, 0) (side B)
        _, _, _, _ = self.env.step(action)
        self.assertEqual(self.env.board[7, 0], 2, "Permanent piece on side B should be opponent's color")

    def test_diagonal_path(self):
        logger.info("Testing diagonal path")
        for i in range(8):
            self.env.board[i, i] = 1
        self.assertTrue(self.env.check_win(), "Player should win with a diagonal path")

    def test_zigzag_path(self):
        logger.info("Testing zigzag path")
        path = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0), (5, 1), (6, 0), (7, 1)]
        for row, col in path:
            self.env.board[row, col] = 1
        self.assertTrue(self.env.check_win(), "Player should win with a zigzag path")

    def test_valid_actions(self):
        logger.info("Testing valid actions")
        valid_actions = self.env.get_valid_actions()
        self.assertEqual(len(valid_actions), self.env.board_width * self.env.board_height * 2, "All actions should be valid on an empty board")

        self.env.board[0, 0] = 1
        valid_actions = self.env.get_valid_actions()
        self.assertEqual(len(valid_actions), self.env.board_width * self.env.board_height * 2 - 1, "One less action should be available after a regular move")

        self.env.board[0, 1] = 2  # Place a permanent piece
        valid_actions = self.env.get_valid_actions()
        self.assertEqual(len(valid_actions), self.env.board_width * self.env.board_height * 2 - 2, "Two less actions should be available after a permanent move")

if __name__ == "__main__":
    unittest.main()

def create_suite():
    logger.info("Creating test suite for BoardGameEnv")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBoardGameEnv))
    return suite
