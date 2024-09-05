import random
import math

class GameBot:
    def __init__(self, board_size=8):
        self.board_size = board_size
        self.board = None
        self.current_player = 1  # 1 for player, -1 for bot
        self.initialize_board()

    def initialize_board(self):
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

    def get_moves(self):
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] == 0]

    def make_move(self, move):
        i, j = move
        if self.board[i][j] == 0:
            self.board[i][j] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        return False

    def undo_move(self, move):
        i, j = move
        self.board[i][j] = 0
        self.current_player *= -1  # Switch player back

    def evaluate(self):
        return sum(sum(row) for row in self.board)

    def alpha_beta_pruning(self, depth, alpha, beta, maximizing_player):
        if depth == 0:
            return self.evaluate()

        moves = self.get_moves()

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                self.make_move(move)
                eval = self.alpha_beta_pruning(depth - 1, alpha, beta, False)
                self.undo_move(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in moves:
                self.make_move(move)
                eval = self.alpha_beta_pruning(depth - 1, alpha, beta, True)
                self.undo_move(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, depth=3):
        best_move = None
        best_eval = -math.inf
        alpha = -math.inf
        beta = math.inf
        moves = self.get_moves()

        for move in moves:
            self.make_move(move)
            eval = self.alpha_beta_pruning(depth - 1, alpha, beta, False)
            self.undo_move(move)

            if eval > best_eval:
                best_eval = eval
                best_move = move

            alpha = max(alpha, eval)

        return best_move  # This will now return a tuple (row, col)

    def train(self, num_games=1000):
        wins = 0
        for _ in range(num_games):
            self.board = self.initialize_board()
            self.current_player = 1

            while self.get_moves():
                move = self.get_best_move() if self.current_player == 1 else self.get_random_move()
                self.make_move(move)

                if self.check_win():
                    if self.current_player == -1:
                        wins += 1
                    break

        return wins / num_games

    def get_random_move(self):
        moves = self.get_moves()
        return random.choice(moves)

    def check_win(self):
        # Implement win condition check
        # This is a placeholder and should be implemented based on your game rules
        return False

    def get_board(self):
        return self.board

    def get_current_player(self):
        return self.current_player

# Initialize the bot
bot = GameBot()
