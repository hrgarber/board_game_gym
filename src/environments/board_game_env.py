import gym
import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))


class BoardGameEnv(gym.Env):
    def __init__(self, board_size=8):
        super().__init__()
        self.board_size = board_size
        self.board = None
        self.current_player = None
        self.action_space = gym.spaces.Discrete(board_size * board_size)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(board_size, board_size), dtype=np.int8
        )
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0:
            return self.board.flatten(), -10, True, {}

        self.board[row, col] = self.current_player

        if self.check_win(row, col):
            return self.board.flatten(), 10, True, {}

        if self.is_draw():
            return self.board.flatten(), 0, True, {}

        if self.check_blocking_move(row, col):
            reward = 1.0
        elif self.check_line(row, col, 4):
            reward = 0.8
        elif self.check_line(row, col, 3):
            reward = 0.5
        else:
            reward = 0.1

        self.current_player *= -1
        return self.board.flatten(), reward, False, {}

    def check_win(self, row, col):
        player = self.board[row, col]
        if player == 0:
            return False
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def check_blocking_move(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # Check if the current move completes the player's own winning line
        if self.check_win(row, col):
            return True

        # Check if the move blocks the opponent's winning move or potential future winning move
        for dr, dc in directions:
            count_same = 1
            count_opponent = 0
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] == self.current_player:
                        count_same += 1
                    elif self.board[r, c] == -self.current_player:
                        count_opponent += 1
                    else:
                        break
                else:
                    break
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.board[r, c] == self.current_player:
                        count_same += 1
                    elif self.board[r, c] == -self.current_player:
                        count_opponent += 1
                    else:
                        break
                else:
                    break

            # If the move blocks an opponent's winning move or creates a winning opportunity
            if count_opponent >= 3 or count_same >= 4:
                return True

        # Check if the move blocks a potential future winning move for the opponent
        original_value = self.board[row, col]
        self.board[row, col] = -self.current_player  # Temporarily place opponent's piece
        if self.check_win(row, col):
            self.board[row, col] = original_value  # Reset the board
            return True
        self.board[row, col] = original_value  # Reset the board

        return False

    def check_line(self, row, col, line_length):
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, line_length):
                r, c = row + i * dr, col + i * dc
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                else:
                    break
            for i in range(1, line_length):
                r, c = row - i * dr, col - i * dc
                if (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                else:
                    break
            if count >= line_length:
                return True
        return False

    def get_state(self):
        return self.board.flatten()

    def render(self, mode="human"):
        board_str = "  " + " ".join([str(i) for i in range(self.board_size)]) + "\n"
        for i, row in enumerate(self.board):
            board_str += f"{i} {' '.join(['X' if cell == 1 else 'O' if cell == -1 else '.' for cell in row])}\n"
        board_str += f"Current player: {'X' if self.current_player == 1 else 'O'}"

        if mode == "human":
            print(board_str)
        return board_str

    def get_valid_actions(self):
        return [i for i, cell in enumerate(self.board.flatten()) if cell == 0]
