# Board Game AI with OpenAI Gym and PyTorch

This project implements a board game AI using OpenAI Gym and PyTorch. The AI can be trained using either Q-learning or Deep Q-Network (DQN) to play a simple board game (similar to Tic-Tac-Toe or Connect Four). The project includes progress evaluation across multiple training sessions, with GPU acceleration support for Windows users with CUDA-enabled devices.

## Project Overview

The project consists of several components:
- A custom OpenAI Gym environment for the board game
- Q-learning and DQN agents implemented with PyTorch for training and decision-making
- Jupyter notebooks for training the AI using both methods and visualizing the learning process
- A command-line interface for playing against the trained AI
- Hyperparameter tuning using grid search, random search, and Bayesian optimization
- A basic web interface for playing the game

## File Structure

```
.
├── src/
│   ├── environments/
│   │   └── board_game_env.py
│   ├── agents/
│   │   ├── q_learning_agent.py
│   │   └── dqn_agent.py
│   └── utils/
│       ├── utils.py
│       ├── hyperparameter_tuning.py
│       ├── agent_evaluation.py
│       └── training_utils.py
├── game_files/
├── notebooks/
│   ├── train_q_learning_ai.ipynb
│   ├── train_dqn_ai.ipynb
│   └── hyperparameter_tuning.ipynb
├── tests/
├── models/
├── config/
├── logs/
├── docs/
├── scripts/
├── main.py
└── requirements.txt
```

## Installation

You can install this project using either conda or pip. Choose the method that best suits your setup.

### Using conda

1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Create a conda environment:
   ```
   conda create --name board_game_gym python=3.8
   ```
4. Activate the conda environment:
   ```
   conda activate board_game_gym
   ```
5. Install the required packages:
   ```
   conda install --file requirements.txt
   ```

### Using pip

1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```
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
   jupyter notebook notebooks/train_q_learning_ai.ipynb
   ```
2. Follow the instructions in the notebook to train the AI using Q-learning.

#### Deep Q-Network (DQN)

1. Open the `train_dqn_ai.ipynb` notebook in Jupyter:
   ```
   jupyter notebook notebooks/train_dqn_ai.ipynb
   ```
2. Follow the instructions in the notebook to train the AI using DQN.

### Hyperparameter Tuning

Use the `hyperparameter_tuning.ipynb` notebook or the functions in `src/utils/hyperparameter_tuning.py` for hyperparameter tuning.

### Playing the Game

1. Run the main script:
   ```
   python main.py --agent [q_learning/dqn] --model [path_to_model_file]
   ```
2. Follow the prompts to make moves and interact with the AI.

### Web Interface

Open the `game_files/index.html` file in a web browser to play the game using the web interface.

## Running Tests

To run the tests for this project, use the following command:

```
python -m unittest discover tests
```

## Contributing

Contributions to improve the project are welcome. Please feel free to submit issues or pull requests.

## License

This project is open-source and available under the MIT License.
