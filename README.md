# Board Game Gym

## Overview

Board Game Gym is a reinforcement learning environment for training AI agents to play board games. This project combines the power of OpenAI Gym-style environments with deep reinforcement learning techniques to create intelligent game-playing agents.

## Features

- Custom board game environment compatible with OpenAI Gym interface
- Implementation of Q-Learning and Deep Q-Network (DQN) agents
- Flexible and extensible architecture for adding new games and agents
- Comprehensive test suite for ensuring code reliability
- Hyperparameter tuning utilities for optimizing agent performance

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
   pip install -r scripts/requirements.txt
   ```

## Project Structure

```
board_game_gym/
├── README.md
├── config/
│   ├── config.py
│   └── pytest.ini
├── docs/
│   └── README.md
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

## Usage

### Training Agents

To train the agents, use the Jupyter notebooks provided in the `notebooks/` directory:

- `train_q_learning_ai.ipynb`: For training the Q-learning agent
- `train_dqn_ai.ipynb`: For training the DQN agent

### Hyperparameter Tuning

Use the `hyperparameter_tuning.ipynb` notebook or the functions in `src/utils/hyperparameter_tuning.py` to optimize agent performance.

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

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and write tests if applicable
4. Run the test suite to ensure everything is working
5. Submit a pull request with a clear description of your changes

For more details, please see the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Gym for the environment interface inspiration
- PyTorch for the deep learning framework
- All contributors who have helped shape this project
