# Board Game AI

This project implements an AI for playing a board game using reinforcement learning techniques.

## Project Structure

```
board_game_ai/
│
├── src/
│   ├── environments/
│   │   └── board_game_env.py
│   ├── agents/
│   │   ├── dqn_agent.py
│   │   └── q_learning_agent.py
│   └── utils/
│       └── utils.py
│
├── tests/
│   ├── test_board_game.py
│   ├── test_dqn_agent.py
│   └── test_training.py
│
├── notebooks/
│   ├── train_board_game_ai.ipynb
│   ├── train_dqn_ai.ipynb
│   └── train_q_learning_ai.ipynb
│
├── game_files/
│   ├── gameBot.js
│   ├── game_bot.py
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Create the conda environment "pathz" if it doesn't exist:
   ```
   conda create -n pathz python=3.9
   ```
4. Activate the conda environment:
   ```
   conda activate pathz
   ```
5. Install the required packages:
   ```
   conda install --file requirements.txt
   ```
   Note: Some packages might need to be installed via pip if not available in conda:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. To train the AI:
   - Use the Jupyter notebooks in the `notebooks/` directory.

2. To play against the trained AI:
   - Run `python main.py`

3. To run tests:
   - Execute `python -m unittest discover tests`

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
