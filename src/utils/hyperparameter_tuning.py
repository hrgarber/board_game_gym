import itertools
import logging
import os
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import optuna
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.environments.board_game_env import BoardGameEnv
from src.utils.utils import evaluate_agent

# Set up logging
log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(
        log_dir, f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set up logging
log_dir = os.path.join("logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(
        log_dir, f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def cross_validate(
    agent_type: str,
    params: Dict[str, Union[float, int]],
    n_splits: int = 5,
    num_episodes: int = 1000,
    eval_episodes: int = 100,
) -> float:
    """
    Perform cross-validation for hyperparameter tuning.

    Args:
        agent_type: Type of agent ('q_learning' or 'dqn').
        params: Dictionary of hyperparameters.
        n_splits: Number of splits for cross-validation.
        num_episodes: Number of episodes to train for each fold.
        eval_episodes: Number of episodes to evaluate each trained agent.

    Returns:
        Mean performance across all folds.
    """
    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    performances = []

    for train_index, _ in kf.split(range(num_episodes)):
        if agent_type == "q_learning":
            agent = QLearningAgent(state_size, action_size, **params)
        elif agent_type == "dqn":
            agent = DQNAgent(state_size, action_size, **params)
        else:
            raise ValueError("Invalid agent type. Choose 'q_learning' or 'dqn'.")

        # Train the agent
        for episode in train_index:
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state

        # Evaluate the agent
        performance = evaluate_agent(env, agent, num_episodes=eval_episodes)
        performances.append(performance)

    return np.mean(performances)


def grid_search(
    agent_type: str,
    param_grid: Dict[str, List[Union[float, int]]],
    num_episodes: int = 1000,
    eval_episodes: int = 100,
    n_splits: int = 5,
) -> Dict[str, Union[List[Dict[str, Union[float, int]]], List[float], Dict[str, Union[float, int]], float]]:
    """
    Perform grid search for hyperparameter tuning with cross-validation.

    Args:
        agent_type: Type of agent ('q_learning' or 'dqn').
        param_grid: Dictionary of parameters and their possible values.
        num_episodes: Number of episodes to train for each combination.
        eval_episodes: Number of episodes to evaluate each trained agent.
        n_splits: Number of splits for cross-validation.

    Returns:
        Dictionary containing results of the grid search.
    """
    logging.info(f"Starting grid search for {agent_type} agent")
    results = {"params": [], "performances": []}

    param_combinations = list(itertools.product(*param_grid.values()))
    for params in tqdm(param_combinations, desc="Grid Search Progress"):
        param_dict = dict(zip(param_grid.keys(), params))
        performance = cross_validate(
            agent_type, param_dict, n_splits, num_episodes, eval_episodes
        )
        results["params"].append(param_dict)
        results["performances"].append(performance)

    best_idx = np.argmax(results["performances"])
    results["best_params"] = results["params"][best_idx]
    results["best_performance"] = results["performances"][best_idx]

    logging.info(
        f"Grid search completed. Best parameters: {results['best_params']}, Best performance: {results['best_performance']}"
    )
    return results


def random_search(
    agent_type: str,
    param_ranges: Dict[str, tuple],
    num_iterations: int = 100,
    num_episodes: int = 1000,
    eval_episodes: int = 100,
    n_splits: int = 5,
) -> Dict[str, Union[List[Dict[str, Union[float, int]]], List[float], Dict[str, Union[float, int]], float]]:
    """
    Perform random search for hyperparameter tuning with cross-validation.

    Args:
        agent_type: Type of agent ('q_learning' or 'dqn').
        param_ranges: Dictionary of parameters and their possible ranges.
        num_iterations: Number of random combinations to try.
        num_episodes: Number of episodes to train for each combination.
        eval_episodes: Number of episodes to evaluate each trained agent.
        n_splits: Number of splits for cross-validation.

    Returns:
        Results of the random search, including all parameter combinations and their performances.
    """
    logging.info(f"Starting random search for {agent_type} agent")
    results = {"params": [], "performances": []}

    for _ in tqdm(range(num_iterations), desc="Random Search Progress"):
        param_dict = {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges.items()}

        performance = cross_validate(
            agent_type, param_dict, n_splits, num_episodes, eval_episodes
        )

        results["params"].append(param_dict)
        results["performances"].append(performance)
        logging.info(f"Parameters: {param_dict}, Performance: {performance}")

    best_idx = np.argmax(results["performances"])
    results["best_params"] = results["params"][best_idx]
    results["best_performance"] = results["performances"][best_idx]

    logging.info(
        f"Random search completed. Best parameters: {results['best_params']}, Best performance: {results['best_performance']}"
    )
    return results


def bayesian_optimization(
    agent_type: str,
    param_ranges: Dict[str, tuple],
    n_trials: int = 100,
    num_episodes: int = 1000,
    eval_episodes: int = 100,
    n_splits: int = 5,
) -> Dict[str, Union[optuna.Study, Dict[str, Union[float, int]], float]]:
    """
    Perform Bayesian optimization for hyperparameter tuning using Optuna with cross-validation.

    Args:
        agent_type: Type of agent ('q_learning' or 'dqn').
        param_ranges: Dictionary of parameters and their possible ranges.
        n_trials: Number of trials for optimization.
        num_episodes: Number of episodes to train for each trial.
        eval_episodes: Number of episodes to evaluate each trained agent.
        n_splits: Number of splits for cross-validation.

    Returns:
        Results of the Bayesian optimization, including the study object and best parameters.
    """
    logging.info(f"Starting Bayesian optimization for {agent_type} agent")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                param_ranges["learning_rate"][0],
                param_ranges["learning_rate"][1],
                log=True,
            ),
            "discount_factor": trial.suggest_float(
                "discount_factor",
                param_ranges["discount_factor"][0],
                param_ranges["discount_factor"][1],
            ),
            "epsilon": trial.suggest_float(
                "epsilon", param_ranges["epsilon"][0], param_ranges["epsilon"][1]
            ),
            "epsilon_decay": trial.suggest_float(
                "epsilon_decay",
                param_ranges["epsilon_decay"][0],
                param_ranges["epsilon_decay"][1],
            ),
        }

        if agent_type == "dqn":
            params["batch_size"] = trial.suggest_int(
                "batch_size",
                param_ranges["batch_size"][0],
                param_ranges["batch_size"][1],
            )

        performance = cross_validate(
            agent_type, params, n_splits, num_episodes, eval_episodes
        )
        logging.info(
            f"Trial {trial.number}: Parameters: {params}, Performance: {performance}"
        )
        return performance

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logging.info(
        f"Bayesian optimization completed. Best parameters: {study.best_params}, Best performance: {study.best_value}"
    )
    return {
        "study": study,
        "best_params": study.best_params,
        "best_performance": study.best_value,
    }
