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
├── requirements.txt
├── .gitignore
├── .pylintrc
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
│   │   └── environments/
│   │       ├── __init__.py
│   │       └── board_game_env.py
│   ├── ai/
│   │   ├── train.py
│   │   └── evaluate.py
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
  - `backend/`: Contains the main Python backend code.
  - `ai/`: Contains scripts for training and evaluating AI models.
  - `tests/`: Holds all test files for the project.
  - `scripts/`: Includes utility scripts for running tests and formatting code.

## Usage

To run the board game environment:

```
python src/backend/main.py
```

To train the AI model:

```
python src/ai/train.py
```

To evaluate the AI model:

```
python src/ai/evaluate.py
```

## Running Tests

To run all tests:

```
python src/tests/run_tests.py
```

## Code Formatting

This project uses Black for code formatting. To format all Python files, run:

```
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
