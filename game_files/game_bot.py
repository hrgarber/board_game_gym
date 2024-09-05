import random
import math

class GameBot:
    """
    A bot that plays the board game using alpha-beta pruning algorithm.
    """

    def __init__(self, board_size=8):
        """
        Initialize the GameBot.

        Args:
            board_size (int): The size of the game board (default is 8x8).
        """
        self.board_size = board_size
        self.board = None
        self.current_player = 1  # 1 for player, -1 for bot
        self.initialize_board()

    def initialize_board(self):
        """Initialize an empty game board."""
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]

    def get_moves(self):
        """Get all possible moves on the current board."""
        return [(i, j) for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] == 0]

    def make_move(self, move):
        """
        Make a move on the board.

        Args:
            move (tuple): The (row, col) position to make the move.

        Returns:
            bool: True if the move was successful, False otherwise.
        """
        i, j = move
        if self.board[i][j] == 0:
            self.board[i][j] = self.current_player
            self.current_player *= -1  # Switch player
            return True
        return False

    def undo_move(self, move):
        """
        Undo a move on the board.

        Args:
            move (tuple): The (row, col) position to undo.
        """
        i, j = move
        self.board[i][j] = 0
        self.current_player *= -1  # Switch player back

    def evaluate(self):
        """Evaluate the current board state."""
        return sum(sum(row) for row in self.board)

    def alpha_beta_pruning(self, depth, alpha, beta, maximizing_player):
        """
        Perform alpha-beta pruning to find the best move.

        Args:
            depth (int): The current depth in the game tree.
            alpha (float): The alpha value for pruning.
            beta (float): The beta value for pruning.
            maximizing_player (bool): True if maximizing, False if minimizing.

        Returns:
            float: The evaluation score of the best move.
        """
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
        """
        Get the best move using alpha-beta pruning.

        Args:
            depth (int): The maximum depth to search in the game tree.

        Returns:
            tuple: The best move as (row, col).
        """
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
        """
        Train the bot by playing against itself.

        Args:
            num_games (int): The number of games to play for training.

        Returns:
            float: The win rate of the bot.
        """
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
        """Get a random valid move."""
        moves = self.get_moves()
        return random.choice(moves)

    def check_win(self):
        """
        Check if the game has been won.

        Note: This is a placeholder and should be implemented based on your game rules.

        Returns:
            bool: True if the game has been won, False otherwise.
        """
        return False

    def get_board(self):
        """Get the current game board."""
        return self.board

    def get_current_player(self):
        """Get the current player."""
        return self.current_player

# Initialize the bot
bot = GameBot()
