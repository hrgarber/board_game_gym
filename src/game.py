import numpy as np


class BoardGame:
    def __init__(self, size=3):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def check_winner(self):
        # Check rows, columns, and diagonals
        for i in range(self.size):
            if abs(np.sum(self.board[i, :])) == self.size:
                return self.board[i, 0]
            if abs(np.sum(self.board[:, i])) == self.size:
                return self.board[0, i]

        if abs(np.trace(self.board)) == self.size:
            return self.board[0, 0]
        if abs(np.trace(np.fliplr(self.board))) == self.size:
            return self.board[0, self.size - 1]

        return 0  # No winner yet

    def is_full(self):
        return np.all(self.board != 0)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1

    def __str__(self):
        return str(self.board)
