import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from environments.board_game_env import BoardGameEnv
from utils.utils import evaluate_agent

# Set up logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "hyperparameter_tuning.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def cross_validate(
    agent_type, params, n_splits=5, num_episodes=1000, eval_episodes=100
):
    """
    Perform cross-validation for hyperparameter tuning.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        params (dict): Dictionary of hyperparameters.
        n_splits (int): Number of splits for cross-validation.
        num_episodes (int): Number of episodes to train for each fold.
        eval_episodes (int): Number of episodes to evaluate each trained agent.

    Returns:
        float: Mean performance across all folds.
    """
    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    performances = []

    for train_index, val_index in kf.split(range(num_episodes)):
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
    agent_type, param_grid, num_episodes=1000, eval_episodes=100, n_splits=5
):
    """
    Perform grid search for hyperparameter tuning with cross-validation.
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
    agent_type,
    param_ranges,
    num_iterations=100,
    num_episodes=1000,
    eval_episodes=100,
    n_splits=5,
):
    """
    Perform random search for hyperparameter tuning with cross-validation.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_ranges (dict): Dictionary of parameters and their possible ranges.
        num_iterations (int): Number of random combinations to try.
        num_episodes (int): Number of episodes to train for each combination.
        eval_episodes (int): Number of episodes to evaluate each trained agent.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        dict: Results of the random search, including all parameter combinations and their performances.
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
    agent_type,
    param_ranges,
    n_trials=100,
    num_episodes=1000,
    eval_episodes=100,
    n_splits=5,
):
    """
    Perform Bayesian optimization for hyperparameter tuning using Optuna with cross-validation.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_ranges (dict): Dictionary of parameters and their possible ranges.
        n_trials (int): Number of trials for optimization.
        num_episodes (int): Number of episodes to train for each trial.
        eval_episodes (int): Number of episodes to evaluate each trained agent.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        dict: Results of the Bayesian optimization, including the study object and best parameters.
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


def visualize_tuning_results(results, method):
    """
    Visualize the results of hyperparameter tuning.

    Args:
        results (dict): Dictionary containing tuning results.
        method (str): The tuning method used ('grid', 'random', or 'bayesian').
    """
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")  # Use non-interactive backend
    import pandas as pd
    import seaborn as sns

    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")

    if method in ["grid", "random"]:
        if "params" in results and "performances" in results:
            df = pd.DataFrame(results["params"])
            df["performance"] = results["performances"]
            df = df.sort_values("performance", ascending=False)

            # Plot performance distribution
            plt.subplot(2, 2, 1)
            sns.histplot(df["performance"], kde=True)
            plt.title(f"{method.capitalize()} Search Performance Distribution")
            plt.xlabel("Performance")

            # Plot top 10 performances
            plt.subplot(2, 2, 2)
            top_10 = df.head(10)
            sns.barplot(x=top_10.index, y="performance", data=top_10)
            plt.title(f"Top 10 {method.capitalize()} Search Results")
            plt.xlabel("Hyperparameter Set")
            plt.ylabel("Performance")

            # Plot parameter correlations
            plt.subplot(2, 2, 3)
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            plt.title("Parameter Correlations")

            # Plot parallel coordinates for top 10 sets
            plt.subplot(2, 2, 4)
            pd.plotting.parallel_coordinates(top_10, "performance")
            plt.title("Top 10 Hyperparameter Sets")
            plt.xticks(rotation=45)

        else:
            print(
                f"Error: 'params' or 'performances' not found in results for {method} search."
            )
            return

    elif method == "bayesian":
        if "study" in results and results["study"] is not None:
            study = results["study"]
            if len(study.trials) > 0:
                # Plot optimization history
                plt.subplot(2, 2, 1)
                optuna.visualization.plot_optimization_history(study)
                plt.title("Bayesian Optimization History")

                # Plot parameter importances
                plt.subplot(2, 2, 2)
                optuna.visualization.plot_param_importances(study)
                plt.title("Parameter Importances")

                # Plot parallel coordinates
                plt.subplot(2, 2, 3)
                optuna.visualization.plot_parallel_coordinate(study)
                plt.title("Parallel Coordinate Plot")

                # Plot slice plot
                plt.subplot(2, 2, 4)
                optuna.visualization.plot_slice(study)
                plt.title("Slice Plot")
            else:
                print("Error: No trials found in the study for Bayesian optimization.")
                return
        else:
            print(
                "Error: 'study' not found or is None in results for Bayesian optimization."
            )
            return
    else:
        print(f"Error: Unknown method '{method}'. Use 'grid', 'random', or 'bayesian'.")
        return

    plt.tight_layout()
    output_dir = os.path.join(project_root, "output", "hyperparameter_tuning")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{method}_tuning_results.png"))
    plt.close()


if __name__ == "__main__":
    q_learning_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "discount_factor": [0.9, 0.95, 0.99],
        "epsilon": [0.1, 0.2, 0.3],
        "epsilon_decay": [0.99, 0.995, 0.999],
    }

    dqn_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "discount_factor": [0.9, 0.95, 0.99],
        "epsilon": [0.1, 0.2, 0.3],
        "epsilon_decay": [0.99, 0.995, 0.999],
        "batch_size": [32, 64, 128],
    }

    print("Grid Search for Q-Learning:")
    q_learning_results = grid_search("q_learning", q_learning_param_grid)
    print(q_learning_results)
    visualize_tuning_results(q_learning_results, "grid")

    print("\nGrid Search for DQN:")
    dqn_results = grid_search("dqn", dqn_param_grid)
    print(dqn_results)
    visualize_tuning_results(dqn_results, "grid")

    q_learning_param_ranges = {
        "learning_rate": (0.001, 0.1),
        "discount_factor": (0.9, 0.99),
        "epsilon": (0.1, 0.5),
        "epsilon_decay": (0.99, 0.9999),
    }

    dqn_param_ranges = {
        "learning_rate": (0.001, 0.1),
        "discount_factor": (0.9, 0.99),
        "epsilon": (0.1, 0.5),
        "epsilon_decay": (0.99, 0.9999),
        "batch_size": (32, 256),
    }

    print("\nRandom Search for Q-Learning:")
    q_learning_random_results = random_search("q_learning", q_learning_param_ranges)
    print(q_learning_random_results)
    visualize_tuning_results(q_learning_random_results, "random")

    print("\nRandom Search for DQN:")
    dqn_random_results = random_search("dqn", dqn_param_ranges)
    print(dqn_random_results)
    visualize_tuning_results(dqn_random_results, "random")

    print("\nBayesian Optimization for Q-Learning:")
    q_learning_bayesian_results = bayesian_optimization(
        "q_learning", q_learning_param_ranges
    )
    print(q_learning_bayesian_results)
    visualize_tuning_results(q_learning_bayesian_results, "bayesian")

    print("\nBayesian Optimization for DQN:")
    dqn_bayesian_results = bayesian_optimization("dqn", dqn_param_ranges)
    print(dqn_bayesian_results)
    visualize_tuning_results(dqn_bayesian_results, "bayesian")

if __name__ == "__main__":
    q_learning_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "discount_factor": [0.9, 0.95, 0.99],
        "epsilon": [0.1, 0.2, 0.3],
        "epsilon_decay": [0.99, 0.995, 0.999],
    }

    dqn_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "discount_factor": [0.9, 0.95, 0.99],
        "epsilon": [0.1, 0.2, 0.3],
        "epsilon_decay": [0.99, 0.995, 0.999],
        "batch_size": [32, 64, 128],
    }

    print("Grid Search for Q-Learning:")
    q_learning_results = grid_search("q_learning", q_learning_param_grid)
    print(q_learning_results)
    visualize_tuning_results(q_learning_results, "grid")

    print("\nGrid Search for DQN:")
    dqn_results = grid_search("dqn", dqn_param_grid)
    print(dqn_results)
    visualize_tuning_results(dqn_results, "grid")

    q_learning_param_ranges = {
        "learning_rate": (0.001, 0.1),
        "discount_factor": (0.9, 0.99),
        "epsilon": (0.1, 0.5),
        "epsilon_decay": (0.99, 0.9999),
    }

    dqn_param_ranges = {
        "learning_rate": (0.001, 0.1),
        "discount_factor": (0.9, 0.99),
        "epsilon": (0.1, 0.5),
        "epsilon_decay": (0.99, 0.9999),
        "batch_size": (32, 256),
    }

    print("\nRandom Search for Q-Learning:")
    q_learning_random_results = random_search("q_learning", q_learning_param_ranges)
    print(q_learning_random_results)
    visualize_tuning_results(q_learning_random_results, "random")

    print("\nRandom Search for DQN:")
    dqn_random_results = random_search("dqn", dqn_param_ranges)
    print(dqn_random_results)
    visualize_tuning_results(dqn_random_results, "random")

    print("\nBayesian Optimization for Q-Learning:")
    q_learning_bayesian_results = bayesian_optimization(
        "q_learning", q_learning_param_ranges
    )
    print(q_learning_bayesian_results)
    visualize_tuning_results(q_learning_bayesian_results, "bayesian")

    print("\nBayesian Optimization for DQN:")
    dqn_bayesian_results = bayesian_optimization("dqn", dqn_param_ranges)
    print(dqn_bayesian_results)
    visualize_tuning_results(dqn_bayesian_results, "bayesian")
import itertools
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.visualization
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.environments.board_game_env import BoardGameEnv
from src.utils.utils import evaluate_agent

# Set up logging
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "hyperparameter_tuning.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def cross_validate(
    agent_type, params, n_splits=5, num_episodes=1000, eval_episodes=100
):
    """
    Perform cross-validation for hyperparameter tuning.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        params (dict): Dictionary of hyperparameters.
        n_splits (int): Number of splits for cross-validation.
        num_episodes (int): Number of episodes to train for each fold.
        eval_episodes (int): Number of episodes to evaluate each trained agent.

    Returns:
        float: Mean performance across all folds.
    """
    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    performances = []

    for train_index, val_index in kf.split(range(num_episodes)):
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
    agent_type, param_grid, num_episodes=1000, eval_episodes=100, n_splits=5
):
    """
    Perform grid search for hyperparameter tuning with cross-validation.
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
    agent_type,
    param_ranges,
    num_iterations=100,
    num_episodes=1000,
    eval_episodes=100,
    n_splits=5,
):
    """
    Perform random search for hyperparameter tuning with cross-validation.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_ranges (dict): Dictionary of parameters and their possible ranges.
        num_iterations (int): Number of random combinations to try.
        num_episodes (int): Number of episodes to train for each combination.
        eval_episodes (int): Number of episodes to evaluate each trained agent.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        dict: Results of the random search, including all parameter combinations and their performances.
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
    agent_type,
    param_ranges,
    n_trials=100,
    num_episodes=1000,
    eval_episodes=100,
    n_splits=5,
):
    """
    Perform Bayesian optimization for hyperparameter tuning using Optuna with cross-validation.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_ranges (dict): Dictionary of parameters and their possible ranges.
        n_trials (int): Number of trials for optimization.
        num_episodes (int): Number of episodes to train for each trial.
        eval_episodes (int): Number of episodes to evaluate each trained agent.
        n_splits (int): Number of splits for cross-validation.

    Returns:
        dict: Results of the Bayesian optimization, including the study object and best parameters.
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


def visualize_tuning_results(results, method):
    """
    Visualize the results of hyperparameter tuning.

    Args:
        results (dict): Dictionary containing tuning results.
        method (str): The tuning method used ('grid', 'random', or 'bayesian').
    """
    plt.switch_backend("agg")  # Use non-interactive backend

    plt.figure(figsize=(15, 10))
    sns.set(style="whitegrid")

    if method in ["grid", "random"]:
        if "params" in results and "performances" in results:
            df = pd.DataFrame(results["params"])
            df["performance"] = results["performances"]
            df = df.sort_values("performance", ascending=False)

            # Plot performance distribution
            plt.subplot(2, 2, 1)
            sns.histplot(df["performance"], kde=True)
            plt.title(f"{method.capitalize()} Search Performance Distribution")
            plt.xlabel("Performance")

            # Plot top 10 performances
            plt.subplot(2, 2, 2)
            top_10 = df.head(10)
            sns.barplot(x=top_10.index, y="performance", data=top_10)
            plt.title(f"Top 10 {method.capitalize()} Search Results")
            plt.xlabel("Hyperparameter Set")
            plt.ylabel("Performance")

            # Plot parameter correlations
            plt.subplot(2, 2, 3)
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            plt.title("Parameter Correlations")

            # Plot parallel coordinates for top 10 sets
            plt.subplot(2, 2, 4)
            pd.plotting.parallel_coordinates(top_10, "performance")
            plt.title("Top 10 Hyperparameter Sets")
            plt.xticks(rotation=45)

        else:
            print(
                f"Error: 'params' or 'performances' not found in results for {method} search."
            )
            return

    elif method == "bayesian":
        if "study" in results and results["study"] is not None:
            study = results["study"]
            if len(study.trials) > 0:
                # Plot optimization history
                plt.subplot(2, 2, 1)
                optuna.visualization.plot_optimization_history(study)
                plt.title("Bayesian Optimization History")

                # Plot parameter importances
                plt.subplot(2, 2, 2)
                optuna.visualization.plot_param_importances(study)
                plt.title("Parameter Importances")

                # Plot parallel coordinates
                plt.subplot(2, 2, 3)
                optuna.visualization.plot_parallel_coordinate(study)
                plt.title("Parallel Coordinate Plot")

                # Plot slice plot
                plt.subplot(2, 2, 4)
                optuna.visualization.plot_slice(study)
                plt.title("Slice Plot")
            else:
                print("Error: No trials found in the study for Bayesian optimization.")
                return
        else:
            print(
                "Error: 'study' not found or is None in results for Bayesian optimization."
            )
            return
    else:
        print(f"Error: Unknown method '{method}'. Use 'grid', 'random', or 'bayesian'.")
        return

    plt.tight_layout()
    output_dir = os.path.join(project_root, "output", "hyperparameter_tuning")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{method}_tuning_results.png"))
    plt.close()


if __name__ == "__main__":
    q_learning_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "discount_factor": [0.9, 0.95, 0.99],
        "epsilon": [0.1, 0.2, 0.3],
        "epsilon_decay": [0.99, 0.995, 0.999],
    }

    dqn_param_grid = {
        "learning_rate": [0.001, 0.01, 0.1],
        "discount_factor": [0.9, 0.95, 0.99],
        "epsilon": [0.1, 0.2, 0.3],
        "epsilon_decay": [0.99, 0.995, 0.999],
        "batch_size": [32, 64, 128],
    }

    print("Grid Search for Q-Learning:")
    q_learning_results = grid_search("q_learning", q_learning_param_grid)
    print(q_learning_results)
    visualize_tuning_results(q_learning_results, "grid")

    print("\nGrid Search for DQN:")
    dqn_results = grid_search("dqn", dqn_param_grid)
    print(dqn_results)
    visualize_tuning_results(dqn_results, "grid")

    q_learning_param_ranges = {
        "learning_rate": (0.001, 0.1),
        "discount_factor": (0.9, 0.99),
        "epsilon": (0.1, 0.5),
        "epsilon_decay": (0.99, 0.9999),
    }

    dqn_param_ranges = {
        "learning_rate": (0.001, 0.1),
        "discount_factor": (0.9, 0.99),
        "epsilon": (0.1, 0.5),
        "epsilon_decay": (0.99, 0.9999),
        "batch_size": (32, 256),
    }

    print("\nRandom Search for Q-Learning:")
    q_learning_random_results = random_search("q_learning", q_learning_param_ranges)
    print(q_learning_random_results)
    visualize_tuning_results(q_learning_random_results, "random")

    print("\nRandom Search for DQN:")
    dqn_random_results = random_search("dqn", dqn_param_ranges)
    print(dqn_random_results)
    visualize_tuning_results(dqn_random_results, "random")

    print("\nBayesian Optimization for Q-Learning:")
    q_learning_bayesian_results = bayesian_optimization(
        "q_learning", q_learning_param_ranges
    )
    print(q_learning_bayesian_results)
    visualize_tuning_results(q_learning_bayesian_results, "bayesian")

    print("\nBayesian Optimization for DQN:")
    dqn_bayesian_results = bayesian_optimization("dqn", dqn_param_ranges)
    print(dqn_bayesian_results)
    visualize_tuning_results(dqn_bayesian_results, "bayesian")
