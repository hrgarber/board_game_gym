import os
import torch
import matplotlib.pyplot as plt
from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent

def save_model(agent, filename):
    """
    Save the agent's model to a file.

    Args:
        agent: The agent (either DQNAgent or QLearningAgent) to save.
        filename (str): The name of the file to save the model to.
    """
    if isinstance(agent, DQNAgent):
        torch.save(agent.model.state_dict(), filename)
    else:
        agent.save_model(filename)


def load_latest_model(agent, models_dir):
    """
    Load the latest model for the given agent from the models directory.

    Args:
        agent: The agent (either DQNAgent or QLearningAgent) to load the model for.
        models_dir (str): The directory containing the saved models.
    """
    if isinstance(agent, DQNAgent):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
            agent.model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
            agent.version = int(latest_model.split('_')[2].split('.')[0])
            print(f"Loaded model: {latest_model}")
        else:
            print("No saved models found.")
    else:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
            agent.load_model(os.path.join(models_dir, latest_model))
            print(f"Loaded model: {latest_model}")
        else:
            print("No saved models found.")


def plot_training_results(rewards, win_rates, agent_name):
    """
    Plot the training results for an agent.

    Args:
        rewards (list): List of episode rewards.
        win_rates (list): List of win rates.
        agent_name (str): Name of the agent for the plot title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(rewards)
    ax1.set_title(f'{agent_name} Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    ax2.plot(win_rates)
    ax2.set_title(f'{agent_name} Win Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')

    plt.tight_layout()
    plt.show()


def plot_version_comparison(env, models_dir):
    """
    Plot a comparison of win rates across different versions of DQN and Q-Learning models.

    Args:
        env: The game environment.
        models_dir (str): The directory containing the saved models.
    """
    dqn_model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    q_model_files = [f for f in os.listdir(models_dir) if f.endswith('.json')]
    
    dqn_versions = sorted(set([int(f.split('_')[2].split('.')[0]) for f in dqn_model_files if f.startswith('dqn_model_')]))
    q_versions = sorted(set([int(f.split('_')[2].split('.')[0]) for f in q_model_files if f.startswith('q_learning_model_')]))

    dqn_win_rates = []
    q_win_rates = []

    for version in dqn_versions:
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device=torch.device("cpu"))
        agent.model.load_state_dict(torch.load(os.path.join(models_dir, f'dqn_model_{version}.pth')))
        win_rate = evaluate_agent(env, agent, num_episodes=100)
        dqn_win_rates.append(win_rate)

    for version in q_versions:
        agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)
        agent.load_model(os.path.join(models_dir, f'q_learning_model_{version}.json'))
        win_rate = evaluate_agent(env, agent, num_episodes=100)
        q_win_rates.append(win_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(dqn_versions, dqn_win_rates, marker='o', label='DQN')
    plt.plot(q_versions, q_win_rates, marker='s', label='Q-Learning')
    plt.title('Win Rate Comparison Across Versions')
    plt.xlabel('Version')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.show()


def evaluate_agent(env, agent, num_episodes=100):
    """
    Evaluate an agent's performance over a number of episodes.

    Args:
        env: The game environment.
        agent: The agent to evaluate.
        num_episodes (int): The number of episodes to evaluate over.

    Returns:
        float: The win rate of the agent.
    """
    wins = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            if reward == 1:
                wins += 1
    return wins / num_episodes

