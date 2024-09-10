import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
import torch.optim as optim

from src.environments.board_game_env import BoardGameEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.utils import load_config, save_results

# Add the project root to the Python path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

def train_dqn_agent(env, agent, config):
    optimizer = optim.Adam(agent.q_network.parameters(), lr=config['LEARNING_RATE'])
    criterion = nn.MSELoss()

    for episode in range(config['NUM_EPISODES']):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(agent.memory) > config['BATCH_SIZE']:
                loss = agent.update(optimizer, criterion)

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return agent

def main():
    parser = argparse.ArgumentParser(description="Train DQN agent for Board Game AI")
    parser.add_argument("--config", default="config/config.py", help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    env = BoardGameEnv(board_size=config['BOARD_SIZE'])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size, action_size, config['BATCH_SIZE'], config['DISCOUNT_FACTOR'], config['EXPLORATION_RATE'], device)

    trained_agent = train_dqn_agent(env, agent, config)

    # Save the trained model
    model_path = os.path.join(project_root, "models", "dqn_model.pth")
    torch.save(trained_agent.q_network.state_dict(), model_path)
    print(f"Trained model saved to {model_path}")

if __name__ == "__main__":
    main()
