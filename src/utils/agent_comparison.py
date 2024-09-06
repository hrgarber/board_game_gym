import matplotlib.pyplot as plt

from src.utils.utils import evaluate_agent


def compare_agents(env, q_agent, dqn_agent, num_episodes=100):
    """
    Compare the performance of Q-Learning and DQN agents.

    Args:
        env: The game environment.
        q_agent: The Q-Learning agent.
        dqn_agent: The DQN agent.
        num_episodes (int): Number of episodes to evaluate each agent.

    Returns:
        tuple: (q_win_rate, dqn_win_rate)
    """
    q_win_rate = evaluate_agent(env, q_agent, num_episodes)
    dqn_win_rate = evaluate_agent(env, dqn_agent, num_episodes)

    return q_win_rate, dqn_win_rate


def plot_agent_comparison(q_win_rate, dqn_win_rate):
    """
    Plot a bar chart comparing the win rates of Q-Learning and DQN agents.

    Args:
        q_win_rate (float): Win rate of the Q-Learning agent.
        dqn_win_rate (float): Win rate of the DQN agent.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(["Q-Learning", "DQN"], [q_win_rate, dqn_win_rate])
    plt.ylabel("Win Rate")
    plt.title("Agent Performance Comparison")
    plt.ylim(0, 1)
    for i, v in enumerate([q_win_rate, dqn_win_rate]):
        plt.text(i, v, f"{v:.2%}", ha="center", va="bottom")
    plt.show()
