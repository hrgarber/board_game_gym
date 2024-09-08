# Board Game Gym

This project implements a reinforcement learning environment for a board game using OpenAI Gym, along with Q-Learning and DQN agents.

## Features

- Board game environment compatible with OpenAI Gym
- Q-Learning agent implementation
- Deep Q-Network (DQN) agent implementation
- Hyperparameter tuning utilities

## Setup

1. Clone the repository
2. Install dependencies (requirements.txt file needed)
3. Run the main script or individual components

## Project Structure

- `src/`: Source code for the project
  - `agents/`: Q-Learning and DQN agent implementations
  - `environments/`: Board game environment
  - `utils/`: Utility functions and hyperparameter tuning
- `config/`: Configuration files
- `tests/`: Unit tests

## Usage

(Add usage instructions here)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

(Add license information here)
# Board Game Gym

## Overview

Board Game Gym is a reinforcement learning environment for training AI agents to play board games. This project combines the power of OpenAI Gym-style environments with deep reinforcement learning techniques to create intelligent game-playing agents.

## Features

- Custom board game environment compatible with OpenAI Gym interface
- Implementation of Deep Q-Network (DQN) agent
- Flexible and extensible architecture for adding new games and agents
- Comprehensive test suite for ensuring code reliability

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hrgarber/board_game_gym.git
   cd board_game_gym
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the DQN Agent

To train the DQN agent on the board game environment:

```python
from src.environments.board_game_env import BoardGameEnv
from src.agents.dqn_agent import DQNAgent

# Create the environment
env = BoardGameEnv(board_size=8)

# Initialize the DQN agent
state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

# Train the agent
num_episodes = 1000
max_steps = 100
agent.train(env, num_episodes, max_steps)
```

### Running Tests

To run all tests:

```
python tests/run_tests.py
```

To run specific test suites:

```
python tests/run_tests.py board_game
python tests/run_tests.py dqn_agent
python tests/run_tests.py training
```

## Project Structure

- `src/`
  - `agents/`: Contains implementation of RL agents (e.g., DQN)
  - `environments/`: Defines the board game environment
- `tests/`: Comprehensive test suite
- `config/`: Configuration files for the project
- `.github/workflows/`: GitHub Actions for CI/CD

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Gym for the environment interface inspiration
- PyTorch for the deep learning framework
