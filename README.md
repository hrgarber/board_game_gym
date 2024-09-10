# Board Game Gym

## Overview

Board Game Gym is a simple board game environment built using OpenAI Gym. This project provides a flexible structure for implementing and playing board games.

## Features

- Custom board game environment using OpenAI Gym
- Flexible architecture for adding new games
- Automatic code formatting using Black
- Web-based game interface (experimental)

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
├── game_files/
│   ├── game_bot.py
│   ├── gameBot.js
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── logs/
│   └── error_log.txt
├── scripts/
│   ├── run_black.py
│   └── run_tests.py
├── src/
│   ├── main.py
│   ├── game.py
│   └── environments/
│       ├── __init__.py
│       └── board_game_env.py
└── tests/
    ├── run_tests.py
    ├── test_board_game.py
    └── test_utils.py
```

### Directory Descriptions

- `config/`: Contains configuration files for the project and pytest.
- `docs/`: Holds documentation files including the project roadmap.
- `game_files/`: Contains files for the experimental web-based game interface.
- `logs/`: Stores log files generated during runtime.
- `scripts/`: Includes utility scripts for running tests and formatting code.
- `src/`: Contains the main source code for the project.
- `tests/`: Holds all test files for the project.

## Usage

To run the board game environment:

```
python src/main.py
```

## Running Tests

To run all tests:

```
python tests/run_tests.py
```

## Code Formatting

This project uses Black for code formatting. To format all Python files, run:

```
python scripts/run_black.py
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
