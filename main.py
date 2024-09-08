import argparse
import os
import sys
from typing import Union
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.environments.board_game_env import BoardGameEnv
from src.utils.utils import load_latest_model, evaluate_agent
from src.utils.training_utils import train_agent
from config import DEVICE, MODEL_DIR

Agent = Union[QLearningAgent, DQNAgent]

def play_game(agent: Agent, env: BoardGameEnv) -> None:
    """
    Play a game using the given agent and environment.

    Args:
        agent: The AI agent (either QLearningAgent or DQNAgent).
        env: The game environment (BoardGameEnv).

    Returns:
        None
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

def train_with_progress(env, agent, num_episodes, max_steps, batch_size=None, update_target_every=None):
    """
    Train the agent with a progress bar.

    Args:
        env: The game environment.
        agent: The AI agent to train.
        num_episodes: Number of episodes to train.
        max_steps: Maximum steps per episode.
        batch_size: Batch size for DQN training (ignored for Q-Learning).
        update_target_every: Number of episodes between target network updates for DQN (ignored for Q-Learning).

    Returns:
        tuple: A tuple containing two lists (rewards, win_rates).
    """
    rewards = []
    win_rates = []

    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        episode_reward, episode_win_rate = train_agent(env, agent, 1, max_steps, batch_size, update_target_every)
        rewards.extend(episode_reward)
        win_rates.extend(episode_win_rate)

        if episode % 100 == 0:
            print(f"Episode {episode}, Win Rate: {win_rates[-1]:.2f}, Total Reward: {rewards[-1]}")

    return rewards, win_rates

def main() -> None:
    """
    Main function to set up and run the game.

    This function parses command-line arguments, initializes the game environment
    and AI agent, loads the trained model, and starts the game.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Play the board game against a trained AI or train a new model."
    )
    parser.add_argument(
        "--agent",
        choices=["q_learning", "dqn"],
        default="q_learning",
        help="Choose the agent type",
    )
    parser.add_argument("--model", help="Path to the trained model file")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()

    env = BoardGameEnv()
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    if args.agent == "q_learning":
        agent: Agent = QLearningAgent(state_size, action_size)
    else:  # args.agent == "dqn"
        agent = DQNAgent(state_size, action_size, DEVICE)

    if args.train:
        print(f"Training {args.agent} agent for {args.episodes} episodes...")
        rewards, win_rates = train_with_progress(env, agent, args.episodes, args.max_steps, batch_size=32, update_target_every=100)
        print("Training complete.")
        # Save the trained model
        agent.save(os.path.join(project_root, MODEL_DIR, f"{args.agent}_model.pth"))
    else:
        model_path = args.model or os.path.join(project_root, MODEL_DIR)
        load_latest_model(agent, model_path)
        play_game(agent, env)


if __name__ == "__main__":
    main()
