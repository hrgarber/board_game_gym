# Board Game Gym

## Overview

Board Game Gym is a reinforcement learning environment for training AI agents to play board games. This project combines the power of OpenAI Gym-style environments with deep reinforcement learning techniques to create intelligent game-playing agents.

## Project Components

- **Custom OpenAI Gym Environment**: A flexible board game environment compatible with the OpenAI Gym interface.
- **Q-Learning Agent**: Implementation of a tabular Q-learning algorithm for simpler game scenarios.
- **Deep Q-Network (DQN) Agent**: A more advanced agent using deep learning for complex game states.
- **Hyperparameter Tuning**: Utilities for optimizing agent performance through grid search, random search, and Bayesian optimization.
- **Jupyter Notebooks**: Interactive notebooks for training, visualization, and analysis.
- **Command-line Interface**: For playing against trained AI models.
- **Web Interface**: A basic web-based game interface for human vs. AI gameplay.

## File Structure

```
board_game_gym/
├── README.md
├── config/
│   ├── config.py
│   └── pytest.ini
├── docs/
│   └── README.md (this file)
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

## Installation

For detailed installation instructions, please refer to the main README.md file in the project root directory.

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

## Testing

Run the test suite using:

```
python -m pytest
```

For more specific test runs, use:

```
python tests/run_tests.py [board_game/dqn_agent/training]
```

## Contributing

We welcome contributions! Please see the main README.md for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the LICENSE file in the project root for full license information.
