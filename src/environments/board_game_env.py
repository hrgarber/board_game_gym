import gym
import numpy as np

class BoardGameEnv(gym.Env):
    """
    A custom OpenAI Gym environment for a board game.
    """

    def __init__(self, board_size=8):
        """
        Initialize the board game environment.

        Args:
            board_size (int): The size of the game board (default is 8x8).
        """
        super().__init__()
        self.board_size = board_size
        self.board = None
        self.current_player = None
        self.action_space = gym.spaces.Discrete(board_size * board_size)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=np.int8)
        self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            numpy.array: The initial state of the environment.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        """
        Take a step in the environment by applying the given action.

        Args:
            action (int): The action to take, represented as a flattened index of the board.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0:
            return self.board.flatten(), -10, True, {}

        self.board[row, col] = self.current_player
        
        # Check for win
        if self.check_win(row, col):
            return self.board.flatten(), 10, True, {}
        
        # Check for draw
        if self.is_draw():
            return self.board.flatten(), 0, True, {}
        
        # Check for blocking opponent's winning move
        if self.check_blocking_move(row, col):
            reward = 1.0
        # Check for creating lines
        elif self.check_line(row, col, 4):
            reward = 0.8
        elif self.check_line(row, col, 3):
            reward = 0.5
        else:
            reward = 0.1  # Small positive reward for non-winning moves
        
        self.current_player *= -1
        return self.board.flatten(), reward, False, {}

    def check_win(self, row, col):
        """
        Check if the current move results in a win.

        Args:
            row (int): The row of the last move.
            col (int): The column of the last move.

        Returns:
            bool: True if the game is won, False otherwise.
        """
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

    def is_draw(self):
        """
        Check if the game is a draw (no more empty cells).

        Returns:
            bool: True if the game is a draw, False otherwise.
        """
        return np.all(self.board != 0)

    def check_blocking_move(self, row, col):
        """
        Check if the current move blocks the opponent's winning move or a line of 4.

        Args:
            row (int): The row of the last move.
            col (int): The column of the last move.

        Returns:
            bool: True if the move blocks a winning move or a line of 4, False otherwise.
        """
        # Temporarily change the player and the board
        original_player = self.current_player
        self.current_player *= -1
        original_value = self.board[row, col]
        self.board[row, col] = self.current_player
        
        blocked_win = self.check_win(row, col)
        blocked_line_of_4 = self.check_line(row, col, 4)
        
        # Check for potential winning moves in all directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            for i in range(-4, 1):  # Check 5 positions in each direction
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    if self.check_win(r, c):
                        blocked_win = True
                        break
        
        # Revert the changes
        self.current_player = original_player
        self.board[row, col] = original_value
        
        return blocked_win or blocked_line_of_4

    def check_line(self, row, col, line_length):
        """
        Check if the current move creates a line of the specified length.

        Args:
            row (int): The row of the last move.
            col (int): The column of the last move.
            line_length (int): The length of the line to check for.

        Returns:
            bool: True if a line of the specified length is created, False otherwise.
        """
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, line_length):
                r, c = row + i * dr, col + i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            for i in range(1, line_length):
                r, c = row - i * dr, col - i * dc
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= line_length:
                return True
        return False

    def get_state(self):
        """
        Get the current state of the board.

        Returns:
            numpy.array: The flattened board state.
        """
        return self.board.flatten()

    def render(self, mode='human'):
        """
        Render the current state of the environment.

        Args:
            mode (str): The mode to render with (default is 'human').

        Returns:
            str: A string representation of the board state.
        """
        board_str = "  " + " ".join([str(i) for i in range(self.board_size)]) + "\n"
        for i, row in enumerate(self.board):
            board_str += f"{i} {' '.join(['X' if cell == 1 else 'O' if cell == -1 else '.' for cell in row])}\n"
        board_str += f"Current player: {'X' if self.current_player == 1 else 'O'}"
        
        if mode == 'human':
            print(board_str)
        return board_str

    def get_valid_actions(self):
        """
        Get a list of valid actions (empty cells) on the board.

        Returns:
            list: A list of valid action indices.
        """
        return [i for i, cell in enumerate(self.board.flatten()) if cell == 0]
