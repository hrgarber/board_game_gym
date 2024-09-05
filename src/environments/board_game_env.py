import gym
import numpy as np

class BoardGameEnv(gym.Env):
    def __init__(self, board_size=8):
        super().__init__()
        self.board_size = board_size
        self.board = None
        self.current_player = None
        self.action_space = gym.spaces.Discrete(board_size * board_size)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=np.int8)
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
        done = self.check_win(row, col)
        reward = 1 if done else 0
        self.current_player *= -1
        return self.board.flatten(), reward, done, {}

    def check_win(self, row, col):
        player = self.board[row, col]
        if player == 0:
            return False
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, 5):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def render(self, mode='human'):
        print("  " + " ".join([str(i) for i in range(self.board_size)]))
        for i, row in enumerate(self.board):
            print(f"{i} {' '.join(['X' if cell == 1 else 'O' if cell == -1 else '.' for cell in row])}")
        print(f"Current player: {'X' if self.current_player == 1 else 'O'}")

    def get_valid_actions(self):
        return [i for i, cell in enumerate(self.board.flatten()) if cell == 0]
