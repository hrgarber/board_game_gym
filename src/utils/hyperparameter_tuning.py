import os
import sys
import itertools
import numpy as np
from tqdm import tqdm
import optuna

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.utils import evaluate_agent

def grid_search(agent_type, param_grid, num_episodes=1000, eval_episodes=100):
    """
    Perform grid search for hyperparameter tuning.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_grid (dict): Dictionary of parameters and their possible values.
        num_episodes (int): Number of episodes to train for each combination.
        eval_episodes (int): Number of episodes to evaluate each trained agent.

    Returns:
        dict: Best parameters and their performance.
    """
    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    best_params = None
    best_performance = float('-inf')

    param_combinations = list(itertools.product(*param_grid.values()))
    for params in tqdm(param_combinations, desc="Grid Search Progress"):
        param_dict = dict(zip(param_grid.keys(), params))
        
        if agent_type == 'q_learning':
            agent = QLearningAgent(state_size, action_size, **param_dict)
        elif agent_type == 'dqn':
            agent = DQNAgent(state_size, action_size, **param_dict)
        else:
            raise ValueError("Invalid agent type. Choose 'q_learning' or 'dqn'.")

        # Train the agent
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state

        # Evaluate the agent
        performance = evaluate_agent(env, agent, num_episodes=eval_episodes)

        if performance > best_performance:
            best_performance = performance
            best_params = param_dict

    return {"best_params": best_params, "best_performance": best_performance}

def random_search(agent_type, param_ranges, num_iterations=100, num_episodes=1000, eval_episodes=100):
    """
    Perform random search for hyperparameter tuning.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_ranges (dict): Dictionary of parameters and their possible ranges.
        num_iterations (int): Number of random combinations to try.
        num_episodes (int): Number of episodes to train for each combination.
        eval_episodes (int): Number of episodes to evaluate each trained agent.

    Returns:
        dict: Best parameters and their performance.
    """
    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    best_params = None
    best_performance = float('-inf')

    for _ in tqdm(range(num_iterations), desc="Random Search Progress"):
        param_dict = {k: np.random.uniform(v[0], v[1]) for k, v in param_ranges.items()}
        
        if agent_type == 'q_learning':
            agent = QLearningAgent(state_size, action_size, **param_dict)
        elif agent_type == 'dqn':
            agent = DQNAgent(state_size, action_size, **param_dict)
        else:
            raise ValueError("Invalid agent type. Choose 'q_learning' or 'dqn'.")

        # Train the agent
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state

        # Evaluate the agent
        performance = evaluate_agent(env, agent, num_episodes=eval_episodes)

        if performance > best_performance:
            best_performance = performance
            best_params = param_dict

    return {"best_params": best_params, "best_performance": best_performance}

def bayesian_optimization(agent_type, param_ranges, n_trials=100, num_episodes=1000, eval_episodes=100):
    """
    Perform Bayesian optimization for hyperparameter tuning using Optuna.

    Args:
        agent_type (str): Type of agent ('q_learning' or 'dqn').
        param_ranges (dict): Dictionary of parameters and their possible ranges.
        n_trials (int): Number of trials for optimization.
        num_episodes (int): Number of episodes to train for each trial.
        eval_episodes (int): Number of episodes to evaluate each trained agent.

    Returns:
        dict: Best parameters and their performance.
    """
    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', param_ranges['learning_rate'][0], param_ranges['learning_rate'][1]),
            'discount_factor': trial.suggest_uniform('discount_factor', param_ranges['discount_factor'][0], param_ranges['discount_factor'][1]),
            'epsilon': trial.suggest_uniform('epsilon', param_ranges['epsilon'][0], param_ranges['epsilon'][1]),
            'epsilon_decay': trial.suggest_uniform('epsilon_decay', param_ranges['epsilon_decay'][0], param_ranges['epsilon_decay'][1])
        }
        
        if agent_type == 'dqn':
            params['batch_size'] = trial.suggest_int('batch_size', param_ranges['batch_size'][0], param_ranges['batch_size'][1])

        if agent_type == 'q_learning':
            agent = QLearningAgent(state_size, action_size, **params)
        elif agent_type == 'dqn':
            agent = DQNAgent(state_size, action_size, **params)
        else:
            raise ValueError("Invalid agent type. Choose 'q_learning' or 'dqn'.")

        # Train the agent
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state

        # Evaluate the agent
        performance = evaluate_agent(env, agent, num_episodes=eval_episodes)
        return performance

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return {"best_params": study.best_params, "best_performance": study.best_value}

# Example usage
if __name__ == "__main__":
    q_learning_param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'discount_factor': [0.9, 0.95, 0.99],
        'epsilon': [0.1, 0.2, 0.3],
        'epsilon_decay': [0.99, 0.995, 0.999]
    }

    dqn_param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'discount_factor': [0.9, 0.95, 0.99],
        'epsilon': [0.1, 0.2, 0.3],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'batch_size': [32, 64, 128]
    }

    print("Grid Search for Q-Learning:")
    q_learning_results = grid_search('q_learning', q_learning_param_grid)
    print(q_learning_results)

    print("\nGrid Search for DQN:")
    dqn_results = grid_search('dqn', dqn_param_grid)
    print(dqn_results)

    q_learning_param_ranges = {
        'learning_rate': (0.001, 0.1),
        'discount_factor': (0.9, 0.99),
        'epsilon': (0.1, 0.5),
        'epsilon_decay': (0.99, 0.9999)
    }

    dqn_param_ranges = {
        'learning_rate': (0.001, 0.1),
        'discount_factor': (0.9, 0.99),
        'epsilon': (0.1, 0.5),
        'epsilon_decay': (0.99, 0.9999),
        'batch_size': (32, 256)
    }

    print("\nRandom Search for Q-Learning:")
    q_learning_random_results = random_search('q_learning', q_learning_param_ranges)
    print(q_learning_random_results)

    print("\nRandom Search for DQN:")
    dqn_random_results = random_search('dqn', dqn_param_ranges)
    print(dqn_random_results)

    print("\nBayesian Optimization for Q-Learning:")
    q_learning_bayesian_results = bayesian_optimization('q_learning', q_learning_param_ranges)
    print(q_learning_bayesian_results)

    print("\nBayesian Optimization for DQN:")
    dqn_bayesian_results = bayesian_optimization('dqn', dqn_param_ranges)
    print(dqn_bayesian_results)
