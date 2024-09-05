import os
import torch
import matplotlib.pyplot as plt
from src.agents.dqn_agent import DQNAgent

def save_model(agent, filename):
    torch.save(agent.model.state_dict(), filename)

def load_latest_model(agent, models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if model_files:
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        agent.model.load_state_dict(torch.load(os.path.join(models_dir, latest_model)))
        agent.version = int(latest_model.split('_')[2].split('.')[0])
        print(f"Loaded model: {latest_model}")
    else:
        print("No saved models found.")

def plot_training_results(rewards, win_rates, version):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.plot(rewards)
    ax1.set_title(f'Episode Rewards (Version {version})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')

    ax2.plot(win_rates)
    ax2.set_title(f'Win Rate (Version {version})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')

    plt.tight_layout()
    plt.show()

def plot_version_comparison(env, models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    versions = sorted(set([int(f.split('_')[2].split('.')[0]) for f in model_files]))

    win_rates = []
    for version in versions:
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device=torch.device("cpu"))
        agent.model.load_state_dict(torch.load(os.path.join(models_dir, f'dqn_model_{version}.pth')))
        win_rate = evaluate_agent(env, agent, num_episodes=100)
        win_rates.append(win_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(versions, win_rates, marker='o')
    plt.title('Win Rate Comparison Across Versions')
    plt.xlabel('Version')
    plt.ylabel('Win Rate')
    plt.show()

def evaluate_agent(env, agent, num_episodes=100):
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
