import unittest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

import numpy as np

from board_game_gym.src.environments.board_game_env import BoardGameEnv

class TestBoardGameEnv(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()

    def test_blocking_potential_win_horizontal(self):
        self.env.board[0, :4] = 1
        result = self.env.check_blocking_move(0, 4)
        self.assertTrue(result, f"Failed to block horizontal potential win:\n{self.env.board}")

    def test_blocking_potential_win_vertical(self):
        self.env.board[:4, 0] = 1
        result = self.env.check_blocking_move(4, 0)
        self.assertTrue(result, f"Failed to block vertical potential win:\n{self.env.board}")

    def test_blocking_opponent_potential_win(self):
        self.env.board[0, :4] = -1  # Opponent's pieces
        result = self.env.check_blocking_move(0, 4)
        self.assertTrue(result, f"Failed to block opponent's potential win:\n{self.env.board}")

    def test_not_blocking_move(self):
        result = self.env.check_blocking_move(4, 4)
        self.assertFalse(result, f"Incorrectly identified as blocking move on empty board:\n{self.env.board}")

    def test_completing_own_winning_line(self):
        self.env.board[0, :4] = 1
        self.env.current_player = 1
        result = self.env.check_blocking_move(0, 4)
        self.assertTrue(result, f"Failed to identify completing own winning line:\n{self.env.board}")

    def test_diagonal_blocking_move(self):
        for i in range(4):
            self.env.board[i, i] = 1
        result = self.env.check_blocking_move(4, 4)
        self.assertTrue(result, f"Failed to block diagonal potential win:\n{self.env.board}")

if __name__ == "__main__":
    unittest.main()
