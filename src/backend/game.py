import numpy as np

class BoardGame:
    def __init__(self):
        self.rows, self.cols = 8, 12
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1  # 1 for white, -1 for black
        self.permanent_pieces = np.zeros((self.rows, self.cols), dtype=bool)

    def make_move(self, row, col, is_permanent):
        if self.board[row, col] == 0:
            if is_permanent:
                self.board[row, col] = -self.current_player
                self.permanent_pieces[row, col] = True
                self._flip_adjacent_pieces(row, col)
            else:
                self.board[row, col] = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def _flip_adjacent_pieces(self, row, col):
        for i in range(max(0, row-1), min(self.rows, row+2)):
            for j in range(max(0, col-1), min(self.cols, col+2)):
                if (i, j) != (row, col) and not self.permanent_pieces[i, j]:
                    self.board[i, j] *= -1

    def check_winner(self):
        for player in [1, -1]:
            if self._has_winning_path(player):
                return player
        return 0  # No winner yet

    def _has_winning_path(self, player):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i, j] == player:
                    if self._dfs(i, j, player, set()):
                        return True
        return False

    def _dfs(self, row, col, player, visited):
        if col == self.cols - 1:
            return True
        
        visited.add((row, col))
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.rows and 0 <= new_col < self.cols and
                    (new_row, new_col) not in visited and
                    self.board[new_row, new_col] == player):
                    if self._dfs(new_row, new_col, player, visited):
                        return True
        
        visited.remove((row, col))
        return False

    def is_full(self):
        return np.all(self.board != 0)

    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.permanent_pieces = np.zeros((self.rows, self.cols), dtype=bool)
        self.current_player = 1

    def get_reward(self, player):
        winner = self.check_winner()
        if winner == player:
            return 10
        elif winner == -player:
            return -10
        return 0

    def __str__(self):
        return str(self.board)
