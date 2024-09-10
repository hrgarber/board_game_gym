# Board Game Gym

## Overview

Board Game Gym is a sophisticated board game environment built using OpenAI Gym. This project provides a flexible structure for implementing and playing board games, with a focus on AI training using Stable Baselines.

## Features

- Custom board game environment using OpenAI Gym
- Flexible architecture for adding new games
- AI opponent training using Stable Baselines
- Performance visualization with TensorBoard
- Automatic code formatting using Black
- Web-based game interface (experimental)

## Game Rules

### Objective

- The goal is to build a continuous path across the long side of the board, connecting one end to the other (sides A and B) in your color.

### Setup

- Players: Two players—one plays as white, the other as black.
- Board: The game starts with an empty 12x8 rectangular grid.
- First Move: The white player always goes first.

### Game Play

- Turns: Players take turns placing one piece at a time anywhere on the board.
- Piece Types:
  - Regular Pieces: Placed showing your color.
  - Permanent Pieces: Placed showing your opponent's color. When placed, they flip all directly touching regular pieces (adjacently and diagonally), including those of your own color. Permanent pieces do not flip other permanent pieces.
- Move Choice: On each turn, a player must choose to place either a regular or a permanent piece, but not both.
- Strategic Use of Permanent Pieces: Permanent pieces are placed in your opponent's color, and they flip the adjacent and diagonal regular pieces. The use of permanent pieces is crucial for both advancing your own path and disrupting your opponent's progress.

### Path Building

- Path Formation: A path is created by connecting pieces that are "touching" each other either adjacently or diagonally.
- Inclusive Path: Your path may include any pieces with your color showing on top, including pieces that were flipped by a permanent piece.
- Path Direction: The path can weave back and forth in any direction as long as it connects opposite ends (sides A and B) and the pieces are linked together.
- Path Flexibility: The path can weave in any direction as long as its pieces touch adjacently or diagonally.

### Winning

- Win Condition: The first player to complete a continuous path connecting sides A and B wins.

### Special Rules

- Edge Rule: If you play a permanent piece at either side A or B, it must be placed in your opponent's color.
- Flipping Mechanic: When placing a permanent piece, it flips all touching regular pieces, but not other permanent pieces.
- Permanent Pieces: These pieces remain static and cannot be flipped once placed, making their placement a significant strategic choice.

### Additional Insights

- Simplified Flow: The game flow is simple but strategic. Players alternate placing pieces, and the dynamic nature of flipping pieces creates opportunities for both advancing your path and disrupting your opponent's progress.
- Permanent Pieces and Flipping: Permanent pieces are especially important as they flip adjacent regular pieces, including your own, creating significant changes in board control.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/board_game_gym.git
   cd board_game_gym
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
board_game_gym/
├── README.md
├── requirements.txt
├── .gitignore
├── .pylintrc
├── train_ai.sh
├── config/
│   ├── config.py
│   └── pytest.ini
├── docs/
│   ├── repo_structure.txt
│   └── ROADMAP.md
├── logs/
│   └── error_log.txt
├── src/
│   ├── frontend/
│   │   ├── index.html
│   │   ├── script.js
│   │   ├── styles.css
│   │   └── gameBot.js
│   ├── backend/
│   │   ├── main.py
│   │   ├── game.py
│   │   ├── game_bot.py
│   │   ├── train_ai.py
│   │   └── environments/
│   │       ├── __init__.py
│   │       └── board_game_env.py
│   ├── tests/
│   │   ├── run_tests.py
│   │   ├── test_board_game.py
│   │   └── test_utils.py
│   └── scripts/
│       ├── run_black.py
│       └── run_tests.py
```

### Directory Descriptions

- `config/`: Contains configuration files for the project and pytest.
- `docs/`: Holds documentation files including the project roadmap.
- `logs/`: Stores log files generated during runtime.
- `src/`: Contains the main source code for the project.
  - `frontend/`: Holds files for the experimental web-based game interface.
  - `backend/`: Contains the main Python backend code, including the AI training script.
  - `tests/`: Holds all test files for the project.
  - `scripts/`: Includes utility scripts for running tests and formatting code.

## Usage

To run the board game environment:

```bash
python src/backend/main.py
```

### Training the AI

There are two ways to train the AI model:

1. Using the convenience script (recommended):

```bash
./train_ai.sh [options]
```

This script will activate the conda environment, install required packages, run the training script with the provided options, and deactivate the environment when finished.

2. Manually running the Python script:

```bash
python src/backend/train_ai.py [options]
```

Both methods will train the AI model using Stable Baselines3, save the trained model, and provide a TensorBoard log for visualization of the training progress.

#### Training Options

The training script now supports the following command-line arguments:

- `--timesteps`: Total timesteps for training (default: 100000)
- `--lr`: Learning rate (default: 0.0003)
- `--n_steps`: Number of steps per update (default: 2048)
- `--eval_episodes`: Number of episodes for evaluation (default: 10)
- `--test_episodes`: Number of episodes for testing (default: 5)

Example usage:

```bash
./train_ai.sh --timesteps 200000 --lr 0.0001 --n_steps 1024
```

This will train the AI for 200,000 timesteps with a learning rate of 0.0001 and 1024 steps per update.

### Visualizing Training Progress

To view the TensorBoard logs during or after training, run:

```bash
tensorboard --logdir=./tensorboard_logs
```

Then open a web browser and go to `http://localhost:6006/` to view the TensorBoard dashboard.

## Running Tests

To run all tests:

```bash
python src/tests/run_tests.py
```

## Code Formatting

This project uses Black for code formatting. To format all Python files, run:

```bash
python src/scripts/run_black.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and write tests if applicable
4. Run the test suite to ensure everything is working
5. Format your code using the `run_black.py` script
6. Submit a pull request with a clear description of your changes

## Roadmap

For information about planned features and improvements, please see the [ROADMAP.md](docs/ROADMAP.md) file in the docs directory.

## License

This project is licensed under the MIT License. See the LICENSE file in the project root for full license information.

## Acknowledgments

- OpenAI Gym for the environment design
- Stable Baselines for the reinforcement learning algorithms
- TensorBoard for visualization tools
