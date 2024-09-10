import sys
from pathlib import Path

import gym
import numpy as np

# Add the project root to the Python path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.backend.game import BoardGame
from src.backend.logger import logger


class BoardGameEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = BoardGame()
        self.action_space = gym.spaces.Discrete(
            self.game.rows * self.game.cols * 2
        )  # *2 for regular and permanent pieces
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.game.rows * self.game.cols,), dtype=np.int8
        )
        logger.info(
            f"BoardGameEnv initialized with board size: {self.game.cols}x{self.game.rows}"
        )

    def reset(self):
        self.game.reset()
        logger.info("Game reset")
        return self.game.board.flatten()

    def step(self, action):
        row, col = divmod(action // 2, self.game.cols)
        is_permanent = action % 2 == 1

        current_player = self.game.current_player

        if not self.game.make_move(row, col, is_permanent):
            logger.warning(f"Invalid move attempted at position ({row}, {col})")
            return self.game.board.flatten(), -10, True, {}

        logger.debug(
            f"Player {current_player} placed a "
            f"{'permanent' if is_permanent else 'regular'} piece at ({row}, {col})"
        )

        reward = self.game.get_reward(current_player)
        done = self.game.check_winner() != 0 or self.game.is_full()

        if done:
            if self.game.check_winner() != 0:
                logger.info(f"Player {self.game.check_winner()} wins!")
            else:
                logger.info("Game ended in a draw")

        logger.debug(f"Turn ended. Current player is now {self.game.current_player}")
        return self.game.board.flatten(), reward, done, {}

    def get_state(self):
        return self.game.board.flatten()

    def render(self, mode="human"):
        board_str = (
            "  "
            + " ".join([str(i).zfill(2) for i in range(self.game.cols)])
            + "\n"
        )
        for i, row in enumerate(self.game.board):
            board_str += f"{str(i).zfill(2)} {' '.join(['W' if cell == 1 else 'B' if cell == -1 else '.' for cell in row])}\n"
        board_str += f"Current player: {'White' if self.game.current_player == 1 else 'Black'}"

        if mode == "human":
            print(board_str)
        logger.debug(f"Current board state:\n{board_str}")
        return board_str

    def get_valid_actions(self):
        valid_actions = []
        for i, cell in enumerate(self.game.board.flatten()):
            if cell == 0:
                valid_actions.extend(
                    [i * 2, i * 2 + 1]
                )  # Regular and permanent piece actions
            else:
                valid_actions.append(
                    i * 2 + 1
                )  # Only permanent piece action for occupied cells
        logger.debug(f"Valid actions: {valid_actions}")
        return valid_actions
