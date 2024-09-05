import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class BoardGameEnv(gym.Env):
    """
    Custom OpenAI Gym environment for a board game (e.g., Tic-Tac-Toe or Connect Four).
    This environment simulates a two-player game on a square board.
    """

    def __init__(self, board_size=8):
        """
        Initialize the board game environment.

        Args:
            board_size (int): Size of the game board (board_size x board_size).
        """
        super(BoardGameEnv, self).__init__()
        
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1  # 1 for player, -1 for opponent
        
        # Define action and observation space
        # Actions are integer indices in the range [0, board_size^2 - 1]
        self.action_space = spaces.Discrete(board_size * board_size)
        # Observations are the board state: -1 for opponent, 0 for empty, 1 for player
        self.observation_space = spaces.Box(low=-1, high=1, shape=(board_size, board_size), dtype=int)

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            numpy.array: The initial state of the board.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        return self.board

    def step(self, action):
        """
        Take a step in the environment by making a move on the board.

        Args:
            action (int): The action to take, represented as an integer in [0, board_size^2 - 1].

        Returns:
            tuple: (observation, reward, done, info)
                observation (numpy.array): The new state of the board after the action.
                reward (float): The reward for taking the action.
                done (bool): Whether the episode has ended.
                info (dict): Additional information about the step.
        """
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row, col] != 0:
            return self.board, -10, True, {'invalid_move': True}

        self.board[row, col] = self.current_player
        
        done = self.check_win() or np.all(self.board != 0)
        
        reward = 0
        if done:
            if self.check_win():
                reward = 1 if self.current_player == 1 else -1
            else:
                reward = 0  # Draw

        self.current_player *= -1
        
        return self.board, reward, done, {}

    def check_win(self):
        """
        Check if the current player has won the game.

        Returns:
            bool: True if the current player has won, False otherwise.
        """
        # Check rows, columns, and diagonals
        for i in range(self.board_size):
            if abs(np.sum(self.board[i, :])) == self.board_size or abs(np.sum(self.board[:, i])) == self.board_size:
                return True
        if abs(np.trace(self.board)) == self.board_size or abs(np.trace(np.fliplr(self.board))) == self.board_size:
            return True
        return False

    def render(self, mode='human'):
        """
        Render the current state of the board.

        Args:
            mode (str): The mode to render with. Can be 'human' or 'rgb_array'.

        Returns:
            numpy.array or None: If mode is 'rgb_array', returns the board image as a numpy array.
        """
        if mode == 'human':
            print(self.board)
        elif mode == 'rgb_array':
            return self.get_board_image()
        return self.board

    def get_valid_actions(self):
        """
        Get a list of valid actions (empty spaces on the board).

        Returns:
            numpy.array: An array of valid action indices.
        """
        return np.where(self.board.flatten() == 0)[0]

    def get_board_image(self):
        """
        Generate a Matplotlib figure representing the current board state.

        Returns:
            matplotlib.figure.Figure: A figure object representing the board.
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.board, cmap='coolwarm', vmin=-1, vmax=1)

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, self.board_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.board_size, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add X and O markers
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    ax.text(j, i, 'X', ha='center', va='center', color='white', fontsize=20, fontweight='bold')
                elif self.board[i, j] == -1:
                    ax.text(j, i, 'O', ha='center', va='center', color='black', fontsize=20, fontweight='bold')

        plt.close(fig)
        return fig