import sys
import os
import argparse
import torch

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.utils import load_latest_model

def play_game(agent, env):
    """
    Play a game using the given agent and environment.

    Args:
        agent: The AI agent (either QLearningAgent or DQNAgent).
        env: The game environment (BoardGameEnv).
    """
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    env.render()
    print(f"Game over. Total reward: {total_reward}")

def main():
    """
    Main function to set up and run the game.
    """
    parser = argparse.ArgumentParser(description="Play the board game against a trained AI.")
    parser.add_argument("--agent", choices=["q_learning", "dqn"], default="q_learning", help="Choose the agent type")
    parser.add_argument("--model", help="Path to the trained model file")
    args = parser.parse_args()

    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.agent == "q_learning":
        agent = QLearningAgent(state_size, action_size)
        model_path = args.model or os.path.join(project_root, "models")
        load_latest_model(agent, model_path)
    elif args.agent == "dqn":
        agent = DQNAgent(state_size, action_size, device)
        model_path = args.model or os.path.join(project_root, "models")
        load_latest_model(agent, model_path)

    play_game(agent, env)

if __name__ == "__main__":
    main()
