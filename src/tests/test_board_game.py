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

    def test_blocking_potential_win_horizontal(self):
        logger.info("Testing horizontal blocking move")
        self.env.board[0, :4] = 1
        result = self.env.check_blocking_move(0, 4)
        self.assertTrue(
            result, f"Failed to block horizontal potential win:\n{self.env.board}"
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

    def test_blocking_potential_win_horizontal_edge(self):
        logger.info("Testing horizontal blocking move at edge")
        self.env.board[0, 4:] = 1
        result = self.env.check_blocking_move(0, 3)
        self.assertTrue(
            result,
            f"Failed to block horizontal potential win at edge:\n{self.env.board}",
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

    def test_blocking_potential_win_vertical(self):
        logger.info("Testing vertical blocking move")
        self.env.board[:4, 0] = 1
        result = self.env.check_blocking_move(4, 0)
        self.assertTrue(
            result, f"Failed to block vertical potential win:\n{self.env.board}"
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

    def test_blocking_opponent_potential_win(self):
        logger.info("Testing blocking opponent's potential win")
        self.env.board[0, :4] = -1  # Opponent's pieces
        result = self.env.check_blocking_move(0, 4)
        self.assertTrue(
            result, f"Failed to block opponent's potential win:\n{self.env.board}"
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

    def test_not_blocking_move(self):
        logger.info("Testing non-blocking move")
        result = self.env.check_blocking_move(4, 4)
        self.assertFalse(
            result,
            f"Incorrectly identified as blocking move on empty board:\n{self.env.board}",
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

    def test_completing_own_winning_line(self):
        logger.info("Testing completing own winning line")
        self.env.board[0, :4] = 1
        self.env.current_player = 1
        result = self.env.check_blocking_move(0, 4)
        self.assertTrue(
            result, f"Failed to identify completing own winning line:\n{self.env.board}"
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

    def test_diagonal_blocking_move(self):
        logger.info("Testing diagonal blocking move")
        for i in range(4):
            self.env.board[i, i] = 1
        result = self.env.check_blocking_move(4, 4)
        self.assertTrue(
            result, f"Failed to block diagonal potential win:\n{self.env.board}"
        )
        logger.debug(f"Board state after test:\n{self.env.render(mode='ansi')}")

if __name__ == "__main__":
    unittest.main()

def create_suite():
    logger.info("Creating test suite for BoardGameEnv")
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestBoardGameEnv))
    return suite
