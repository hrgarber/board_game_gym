import sys
from pathlib import Path

import gym
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.backend.logger import logger

class BoardGameEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.board_width = 12
        self.board_height = 8
        self.board = None
        self.current_player = None
        self.action_space = gym.spaces.Discrete(self.board_width * self.board_height * 2)  # *2 for regular and permanent pieces
        self.observation_space = gym.spaces.Box(
            low=-2, high=2, shape=(self.board_height, self.board_width), dtype=np.int8
        )
        self.reset()
        logger.info(f"BoardGameEnv initialized with board size: {self.board_width}x{self.board_height}")

    def reset(self):
        self.board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        self.current_player = 1
        logger.info("Game reset")
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action // 2, self.board_width)
        is_permanent = action % 2 == 1

        if not is_permanent and self.board[row, col] != 0:
            logger.warning(f"Invalid move attempted at position ({row}, {col})")
            return self.board.flatten(), -10, True, {}

        piece_value = self.current_player * (2 if is_permanent else 1)
        
        # Apply edge rule
        if is_permanent and (col == 0 or col == self.board_width - 1):
            piece_value *= -1

        self.board[row, col] = piece_value

        if is_permanent:
            self._flip_adjacent_pieces(row, col)

        logger.debug(f"Player {self.current_player} placed a {'permanent' if is_permanent else 'regular'} piece at ({row}, {col})")

        if self.check_win():
            logger.info(f"Player {self.current_player} wins!")
            return self.board.flatten(), 10, True, {}

        if self.is_draw():
            logger.info("Game ended in a draw")
            return self.board.flatten(), 0, True, {}

        self.current_player *= -1
        logger.debug(f"Turn ended. Current player is now {self.current_player}")
        return self.board.flatten(), 0, False, {}

    def _flip_adjacent_pieces(self, row, col):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.board_height and 0 <= c < self.board_width:
                    if abs(self.board[r, c]) == 1:  # Only flip regular pieces
                        self.board[r, c] *= -1  # Flip to the opposite color

    def check_win(self):
        for player in [1, -1]:
            for start_col in range(self.board_width):
                if self._has_path(player, 0, start_col):
                    return True
        return False

    def _has_path(self, player, row, col):
        if row < 0 or row >= self.board_height or col < 0 or col >= self.board_width:
            return False
        if self.board[row, col] != player and self.board[row, col] != player * 2:
            return False
        if row == self.board_height - 1:
            return True

        original_value = self.board[row, col]
        self.board[row, col] = 0  # Mark as visited
        for dr, dc in [(0, 1), (0, -1), (1, 0), (1, 1), (1, -1)]:
            if self._has_path(player, row + dr, col + dc):
                self.board[row, col] = original_value  # Restore the original value
                return True
        self.board[row, col] = original_value  # Restore the original value
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def get_state(self):
        return self.board.flatten()

    def render(self, mode="human"):
        board_str = "  " + " ".join([str(i).zfill(2) for i in range(self.board_width)]) + "\n"
        for i, row in enumerate(self.board):
            board_str += f"{str(i).zfill(2)} {' '.join(['W' if cell == 1 else 'w' if cell == 2 else 'B' if cell == -1 else 'b' if cell == -2 else '.' for cell in row])}\n"
        board_str += f"Current player: {'White' if self.current_player == 1 else 'Black'}"

        if mode == "human":
            print(board_str)
        logger.debug(f"Current board state:\n{board_str}")
        return board_str

    def get_valid_actions(self):
        valid_actions = []
        for i, cell in enumerate(self.board.flatten()):
            if cell == 0:
                valid_actions.extend([i * 2, i * 2 + 1])  # Regular and permanent piece actions
            else:
                valid_actions.append(i * 2 + 1)  # Only permanent piece action for occupied cells
        logger.debug(f"Valid actions: {valid_actions}")
        return valid_actions
