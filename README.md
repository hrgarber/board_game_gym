# Board Game Gym

## Overview

Board Game Gym is a reinforcement learning environment for training AI agents to play board games. This project combines the power of OpenAI Gym-style environments with deep reinforcement learning techniques to create intelligent game-playing agents.

## Features

- Custom board game environment
- Implementation of AI agents (specific types to be determined)
- Flexible architecture for adding new games and agents
- Test suite for ensuring code reliability

## Project Components

- **Custom OpenAI Gym Environment**: A flexible board game environment compatible with the OpenAI Gym interface.
- **Q-Learning Agent**: Implementation of a tabular Q-learning algorithm for simpler game scenarios.
- **Deep Q-Network (DQN) Agent**: A more advanced agent using deep learning for complex game states.
- **Hyperparameter Tuning**: Utilities for optimizing agent performance through grid search, random search, and Bayesian optimization.
- **Jupyter Notebooks**: Interactive notebooks for training, visualization, and analysis.
- **Command-line Interface**: For playing against trained AI models.
- **Web Interface**: A basic web-based game interface for human vs. AI gameplay.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/board_game_gym.git
   cd board_game_gym
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
board_game_gym/
├── README.md
├── config/
│   ├── config.py
│   └── pytest.ini
├── game_files/
│   └── index.html
├── notebooks/
│   ├── train_q_learning_ai.ipynb
│   ├── train_dqn_ai.ipynb
│   └── hyperparameter_tuning.ipynb
├── scripts/
│   └── requirements.txt
├── src/
│   ├── agents/
│   │   ├── q_learning_agent.py
│   │   └── dqn_agent.py
│   ├── environments/
│   │   └── board_game_env.py
│   └── utils/
│       └── hyperparameter_tuning.py
└── tests/
    ├── test_board_game.py
    ├── test_dqn_agent.py
    └── test_training.py
```

## Game Logic

The `src/environments/board_game_env.py` file contains the core game logic for the board game. It includes:

- A `BoardGameEnv` class that represents the game state and implements the OpenAI Gym interface
- Methods for making moves, checking for winners, and resetting the game

To play the game or integrate it with AI agents, you can import and use the `BoardGameEnv` class from `src/environments/board_game_env.py`.

## Usage

### Training Agents

To train the agents, use the Jupyter notebooks provided in the `notebooks/` directory:

- `train_q_learning_ai.ipynb`: For training the Q-learning agent
- `train_dqn_ai.ipynb`: For training the DQN agent

### Hyperparameter Tuning

Use the `hyperparameter_tuning.ipynb` notebook or the functions in `src/utils/hyperparameter_tuning.py` to optimize agent performance.

### Playing the Game

1. Command-line Interface:
   ```
   python main.py --agent [q_learning/dqn] --model [path_to_model_file]
   ```

2. Web Interface:
   Open `game_files/index.html` in a web browser.

## Running Tests

To run all tests:

```
python -m pytest
```

For more specific test runs, use:

```
python tests/run_tests.py [board_game/dqn_agent/training]
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and write tests if applicable
4. Run the test suite to ensure everything is working
5. Submit a pull request with a clear description of your changes

## License

This project is licensed under the MIT License. See the LICENSE file in the project root for full license information.

## Acknowledgments

- All contributors who have helped shape this project
- OpenAI Gym for inspiration on environment design
- PyTorch community for deep learning resources
