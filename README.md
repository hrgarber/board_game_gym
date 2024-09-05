# Board Game AI with OpenAI Gym and PyTorch

This project implements a board game AI using OpenAI Gym and PyTorch. The AI can be trained using either Q-learning or Deep Q-Network (DQN) to play a simple board game (similar to Tic-Tac-Toe or Connect Four). The project includes progress evaluation across multiple training sessions, with GPU acceleration support for Windows users with CUDA-enabled devices.

## Project Overview

The project consists of several components:
- A custom OpenAI Gym environment for the board game
- Q-learning and DQN agents implemented with PyTorch for training and decision-making
- Jupyter notebooks for training the AI using both methods and visualizing the learning process
- A command-line interface for playing against the trained AI

## Roadmap

### Completed
1. Set up the project structure with necessary directories and files
2. Implemented the core game environment (BoardGameEnv) in Python
3. Created the Q-learning agent (QLearningAgent) in Python
4. Developed the Deep Q-Network agent (DQNAgent) in Python
5. Implemented utility functions for saving/loading models and visualizing results
6. Created Jupyter notebooks for training both Q-learning and DQN agents
7. Implemented a basic web interface for the game using HTML, CSS, and JavaScript
8. Added unit tests for the board game environment, Q-learning agent, and DQN agent
9. Created a main script for playing against the trained AI
10. Implemented Alpha-Beta Pruning algorithm for improved decision-making

### In Progress
11. Refining and optimizing the training process for Q-learning, DQN, and Alpha-Beta Pruning agents

### Upcoming
12. Improve the web interface to allow playing against the trained AI
13. Implement cross-platform compatibility checks and optimizations
14. Add more comprehensive documentation and comments to the code
15. Perform thorough testing and debugging of all components
16. Create a user guide for setting up and using the project
17. Optimize performance for larger board sizes and more complex game rules
18. Implement additional AI algorithms for comparison (e.g., SARSA, A3C)
19. Add support for multiplayer games (AI vs AI, Human vs Human)
20. Develop a graphical user interface (GUI) for easier interaction with the game and AI

## File Structure

- `src/`
  - `environments/`
    - `board_game_env.py`: Contains the custom OpenAI Gym environment for the board game.
  - `agents/`
    - `q_learning_agent.py`: Implements the Q-learning agent using PyTorch.
    - `dqn_agent.py`: Implements the DQN agent using PyTorch.
  - `utils/`
    - `utils.py`: Contains utility functions.
- `game_files/`: Contains game-related files.
- `notebooks/`
  - `train_q_learning_ai.ipynb`: Jupyter notebook for training the AI using Q-learning.
  - `train_dqn_ai.ipynb`: Jupyter notebook for training the AI using DQN.
- `main.py`: Provides a command-line interface for playing against the trained AI.
- `requirements.txt`: Lists the required Python packages.
- `models/`: Directory for storing trained model versions.

## Requirements

- Python 3.7 or later
- OpenAI Gym
- NumPy
- Matplotlib
- Jupyter Notebook
- PyTorch
- tqdm

## Installation

1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```
5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the AI

You have two options for training the AI:

#### Q-learning

1. Open the `train_q_learning_ai.ipynb` notebook in Jupyter:
   ```
   jupyter notebook notebooks/train_q_learning_ai.ipynb
   ```
2. Follow the instructions in the notebook to train the AI using Q-learning.

#### Deep Q-Network (DQN)

1. Open the `train_dqn_ai.ipynb` notebook in Jupyter:
   ```
   jupyter notebook notebooks/train_dqn_ai.ipynb
   ```
2. Follow the instructions in the notebook to train the AI using DQN.

Both notebooks will automatically use GPU acceleration if available on Windows with a CUDA-enabled device. They will visualize the training progress and save the trained models.

Key features of the training process:
- Visualization of rewards and win rates over time
- GPU acceleration support for faster training on Windows with CUDA-enabled devices
- Ability to save and load trained models

### Playing the Game

1. Run the main script:
   ```
   python main.py --agent [q_learning/dqn] --model [path_to_model_file]
   ```
   Replace `[q_learning/dqn]` with your choice of agent and `[path_to_model_file]` with the path to your trained model file.
2. Follow the prompts to make moves and interact with the AI.

## Running Tests

To run the tests for this project, use the following command:

```
python -m unittest discover tests
```

This will run all the tests in the `tests` directory and report the results.

As of the latest update, all 23 tests are passing successfully. The test suite covers various aspects of the project, including the game environment, Q-learning agent, and DQN agent.

## Implementations

### Q-learning

The Q-learning implementation uses a tabular approach with PyTorch tensors:
- Stores Q-values in a PyTorch tensor for efficient updates and GPU acceleration.
- Uses an epsilon-greedy policy for action selection.
- Implements experience replay for improved learning stability.

### Deep Q-Network (DQN)

The DQN implementation uses a neural network to approximate the Q-function:
- Uses a PyTorch neural network for Q-value approximation.
- Implements experience replay and a separate target network for improved stability during training.
- Uses an epsilon-greedy policy with decaying exploration rate.

## Customization

You can customize various aspects of the project:

- Modify the `BoardGameEnv` class in `board_game_env.py` to change the game rules or board size.
- Adjust the hyperparameters in `q_learning_agent.py` and `dqn_agent.py` to experiment with different learning strategies.
- Modify the training process in the Jupyter notebooks to change the number of episodes, batch size, or add different visualizations.
- Experiment with different neural network architectures in the `DQN` class for improved performance.

## Cross-Platform Compatibility

This project is designed to work on both Windows and Mac:
- On Windows with a CUDA-enabled GPU, the training process will automatically use GPU acceleration for faster computations.
- On Mac or Windows without a CUDA-enabled GPU, the training process will run on the CPU.

The code automatically detects the available hardware and adjusts accordingly, ensuring seamless operation across different platforms.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the MIT License.
