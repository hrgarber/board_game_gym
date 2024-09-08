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
import argparse
import json
from typing import Dict, Any

from src.utils.hyperparameter_tuning import (
    bayesian_optimization,
    grid_search,
    random_search,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the tuning configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save the tuning results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Board Game AI"
    )
    parser.add_argument(
        "agent", choices=["q_learning", "dqn"], help="Type of agent to tune"
    )
    parser.add_argument(
        "method", choices=["grid", "random", "bayesian"], help="Tuning method to use"
    )
    parser.add_argument(
        "--config",
        default="tuning_config.json",
        help="Path to tuning configuration file",
    )
    parser.add_argument(
        "--output", default="tuning_results.json", help="Path to save tuning results"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    tuning_methods = {
        "grid": grid_search,
        "random": random_search,
        "bayesian": bayesian_optimization,
    }

    results = tuning_methods[args.method](
        args.agent,
        config[args.agent]["param_grid" if args.method == "grid" else "param_ranges"],
    )

    output_dir = os.path.join(project_root, "output", "hyperparameter_tuning")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output)
    save_results(results, output_file)
    print(f"Tuning completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
