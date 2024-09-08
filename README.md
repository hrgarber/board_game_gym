# Board Game Gym

## Overview

Board Game Gym is a reinforcement learning environment for training AI agents to play board games. This project combines the power of OpenAI Gym-style environments with deep reinforcement learning techniques to create intelligent game-playing agents.

## Rules

### Objective:
- The goal is to build a continuous path across the long side of the board, connecting one end to the other (sides A and B).

### Setup:
- Players: Two players—one plays as white, the other as black.
- Board: The game starts with an empty 12x8 rectangular grid.
- First Move: The white player always goes first.

### Game Play:
- Turns: Players take turns placing one piece at a time anywhere on the board.
- Piece Types:
  - Regular Pieces: Placed showing your color.
  - Permanent Pieces: Placed showing your opponent's color. When placed, they flip all directly touching regular pieces (adjacently and diagonally), including those of your own color. Permanent pieces do not flip other permanent pieces.
- Move Choice: On each turn, a player must choose to place either a regular or a permanent piece, but not both.

### Path Building:
- Path Formation: A path is created by connecting pieces that are "touching" each other either adjacently or diagonally.
- Inclusive Path: Your path may include any pieces with your color showing on top.
- Path Direction: The path can weave back and forth in any direction as long as it connects opposite ends (sides A and B) and the pieces are linked together.

### Winning:
- The game is won by the first player who completes a path connecting sides A and B.

### Special Rules:
- Edge Rule: If you play a permanent piece at either side A or B, it must be placed in your opponent's color.
- Flipping Mechanic: When placing a permanent piece, it flips all touching regular pieces, but not other permanent pieces.

This strategic game revolves around path-building, blocking the opponent, and the clever use of both regular and permanent pieces.

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
│   ├── utils/
│   │   └── hyperparameter_tuning.py
│   └── game.py
└── tests/
    ├── test_board_game.py
    ├── test_dqn_agent.py
    └── test_training.py
```

## Game Logic

The `src/game.py` file contains the core game logic for the board game. It includes:

- A `BoardGame` class that represents the game state
- Methods for making moves, checking for winners, and resetting the game
- A flexible board size (default is 3x3 for Tic-Tac-Toe)

To play the game or integrate it with AI agents, you can import and use the `BoardGame` class from `src/game.py`.

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
