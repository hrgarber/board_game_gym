import numpy as np
import matplotlib.pyplot as plt

def evaluate_agent(env, agent, num_episodes=100, epsilon=0):
    """
    Evaluate an agent's performance in the environment without exploration.
    
    Args:
        env: The environment to evaluate in.
        agent: The agent to evaluate.
        num_episodes: Number of episodes to run for evaluation.
        epsilon: Exploration rate (set to 0 for evaluation).
    
    Returns:
        A dictionary containing evaluation metrics.
    """
    results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'successes': []
    }
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.act(state)  # Assuming act method doesn't use epsilon
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        results['episode_rewards'].append(episode_reward)
        results['episode_lengths'].append(episode_length)
        results['successes'].append(episode_reward > 0)  # Adjust this condition as needed
    
    return results

def calculate_metrics(evaluation_results):
    """
    Calculate performance metrics from evaluation results.
    
    Args:
        evaluation_results: Dictionary containing evaluation data.
    
    Returns:
        A dictionary containing calculated metrics.
    """
    metrics = {
        'average_reward': np.mean(evaluation_results['episode_rewards']),
        'success_rate': np.mean(evaluation_results['successes']),
        'average_episode_length': np.mean(evaluation_results['episode_lengths'])
    }
    return metrics

def plot_learning_curve(episode_rewards, title):
    """
    Plot the learning curve (episode rewards over time).
    
    Args:
        episode_rewards: List of episode rewards.
        title: Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_average_reward(average_rewards, title):
    """
    Plot the average reward per episode.
    
    Args:
        average_rewards: List of average rewards.
        title: Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(average_rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_loss_curve(losses, title):
    """
    Plot the loss curve over time (for DQN).
    
    Args:
        losses: List of loss values.
        title: Title for the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.close()
