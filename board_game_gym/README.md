# Board Game AI with OpenAI Gym and PyTorch

This project implements a board game AI using OpenAI Gym and PyTorch. The AI can be trained using either Q-learning or Deep Q-Network (DQN) to play a simple board game (similar to Tic-Tac-Toe or Connect Four). The project includes progress evaluation across multiple training sessions, with GPU acceleration support for Windows users with CUDA-enabled devices.

## Project Overview

The project consists of several components:
- A custom OpenAI Gym environment for the board game
- Q-learning and DQN agents implemented with PyTorch for training and decision-making
- Jupyter notebooks for training the AI using both methods and visualizing the learning process
- A command-line interface for playing against the trained AI

## File Structure

- `board_game_env.py`: Contains the custom OpenAI Gym environment for the board game.
- `q_learning_agent.py`: Implements the Q-learning agent using PyTorch.
- `dqn_agent.py`: Implements the DQN agent using PyTorch.
- `main.py`: Provides a command-line interface for playing against the trained AI.
- `train_q_learning_ai.ipynb`: Jupyter notebook for training the AI using Q-learning.
- `train_dqn_ai.ipynb`: Jupyter notebook for training the AI using DQN.
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
3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
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
   jupyter notebook train_q_learning_ai.ipynb
   ```
2. Follow the instructions in the notebook to train the AI using Q-learning.

#### Deep Q-Network (DQN)

1. Open the `train_dqn_ai.ipynb` notebook in Jupyter:
   ```
   jupyter notebook train_dqn_ai.ipynb
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
   python main.py
   ```
2. The script will prompt you to choose between the Q-learning and DQN models.
3. Follow the prompts to make moves and interact with the AI.

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
- On Windows with a CUDA-enabled GPU (e.g., RTX 4090), the training process will automatically use GPU acceleration for faster computations.
- On Mac or Windows without a CUDA-enabled GPU, the training process will run on the CPU.

The code automatically detects the available hardware and adjusts accordingly, ensuring seamless operation across different platforms.

## Contributing

Contributions to improve the project are welcome. Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the MIT License.