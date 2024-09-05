import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import logging
import json
import os
from tqdm.notebook import tqdm

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_models_directory():
    """Create a directory for saving models if it doesn't exist."""
    os.makedirs("models", exist_ok=True)

def load_latest_model(agent, directory):
    """
    Load the latest model for the agent.

    Args:
        agent (QLearningAgent): The Q-learning agent.
        directory (str): The directory to search for models.

    Returns:
        bool: True if a model was loaded, False otherwise.
    """
    try:
        if agent.load_latest_model(directory):
            logging.info(f"Loaded model version {agent.version}")
            return True
        else:
            logging.info("No previous model found. Starting from scratch.")
            return False
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON when loading the model: {str(e)}")
        logging.info("Starting from scratch due to model loading error.")
        return False
    except Exception as e:
        logging.error(f"Unexpected error when loading the model: {str(e)}")
        logging.info("Starting from scratch due to unexpected error.")
        return False

def initialize_training(num_episodes):
    """Initialize training variables and progress bar."""
    rewards = []
    win_rates = []
    batch_win_rates = []
    pbar = tqdm(total=num_episodes, desc="Training Progress", position=0, leave=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    return rewards, win_rates, batch_win_rates, pbar, fig, ax1, ax2

def run_episode(env, agent):
    """Run a single episode of the game."""
    state = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        try:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            
            done = bool(done)
        except Exception as e:
            logging.error(f"Error in episode: {str(e)}")
            break
    
    return episode_reward

def update_training_progress(episode, rewards, win_rates, agent, pbar):
    """Update training progress and win rates."""
    if (episode + 1) % 100 == 0:
        win_rate = sum(rewards[-100:]) / 200 + 0.5
        win_rates.append(win_rate)
        agent.update_training_history(episode + 1, win_rate)
        pbar.set_description(f"Training Progress - Win Rate: {win_rate:.2f}")

def calculate_batch_win_rate(rewards, batch_size):
    """Calculate the win rate for a batch of episodes."""
    return sum(rewards[-batch_size:]) / (2 * batch_size) + 0.5

def save_model(agent, save_interval, episode):
    """Save the model at specified intervals."""
    if (episode + 1) % save_interval == 0:
        saved_filename = agent.save_model_with_version("models")
        logging.info(f"Model saved as {saved_filename}")

def train_agent(env, agent, num_episodes, batch_size, save_interval=1000):
    """
    Train the Q-learning agent and visualize the training progress.

    Args:
        env (BoardGameEnv): The game environment.
        agent (QLearningAgent): The Q-learning agent to train.
        num_episodes (int): The total number of episodes to train for.
        batch_size (int): The number of episodes per batch.
        save_interval (int): The interval at which to save the model.

    Returns:
        tuple: (trained_agent, rewards, win_rates, batch_win_rates)
    """
    rewards, win_rates, batch_win_rates, pbar, fig, ax1, ax2 = initialize_training(num_episodes)
    
    for episode in range(num_episodes):
        episode_reward = run_episode(env, agent)
        rewards.append(episode_reward)
        pbar.update(1)
        
        update_training_progress(episode, rewards, win_rates, agent, pbar)
        
        if (episode + 1) % batch_size == 0:
            batch_win_rate = calculate_batch_win_rate(rewards, batch_size)
            batch_win_rates.append(batch_win_rate)
            update_training_plots(ax1, ax2, win_rates, batch_win_rates, agent.version)
        
        save_model(agent, save_interval, episode)
    
    pbar.close()
    plt.close(fig)

    return agent, rewards, win_rates, batch_win_rates

def update_training_plots(ax1, ax2, win_rates, batch_win_rates, agent_version):
    """Update the training progress plots."""
    update_win_rate_plot(ax1, win_rates, agent_version)
    update_batch_win_rate_plot(ax2, batch_win_rates)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

def update_win_rate_plot(ax, win_rates, agent_version):
    """Update the win rate over time plot."""
    ax.clear()
    ax.plot(range(100, len(win_rates) * 100 + 1, 100), win_rates)
    ax.set_title(f"AI Win Rate Over Time (Version {agent_version})")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Win Rate")

def update_batch_win_rate_plot(ax, batch_win_rates):
    """Update the batch win rate plot."""
    ax.clear()
    ax.bar(range(1, len(batch_win_rates) + 1), batch_win_rates)
    ax.set_title("Batch Win Rates")
    ax.set_xlabel("Batch Number")
    ax.set_ylabel("Win Rate")

def plot_final_results(win_rates, batch_win_rates, agent_version):
    """Plot the final training results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    update_win_rate_plot(ax1, win_rates, agent_version)
    update_batch_win_rate_plot(ax2, batch_win_rates)
    plt.tight_layout()
    plt.show()

def plot_version_comparison(env, directory):
    """
    Plot a comparison of win rates across different model versions.
    
    Args:
        env (BoardGameEnv): The game environment.
        directory (str): The directory containing the saved models.
    """
    versions, win_rates = get_version_win_rates(env, directory)
    
    if versions and win_rates:
        plot_version_win_rates(versions, win_rates)
    else:
        print("No valid model versions found for comparison.")

def get_version_win_rates(env, directory):
    """Get the win rates for different versions of the model."""
    versions = []
    win_rates = []
    
    for filename in os.listdir(directory):
        if filename.startswith("model_v") and filename.endswith(".json"):
            version, win_rate = load_model_version(env, directory, filename)
            if version and win_rate is not None:
                versions.append(version)
                win_rates.append(win_rate)
    
    return versions, win_rates

def load_model_version(env, directory, filename):
    """Load a specific model version and return its version and win rate."""
    temp_agent = QLearningAgent(state_size=env.board_size * env.board_size, action_size=env.board_size * env.board_size)
    try:
        temp_agent.load_model(os.path.join(directory, filename))
        return temp_agent.version, temp_agent.training_history[-1]["win_rate"]
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON for file {filename}. Skipping this file.")
    except Exception as e:
        logging.error(f"Unexpected error loading {filename}: {str(e)}. Skipping this file.")
    return None, None

def plot_version_win_rates(versions, win_rates):
    """Plot the win rates for different versions of the model."""
    plt.figure(figsize=(12, 4))
    plt.bar(versions, win_rates)
    plt.title("Win Rate Comparison Across Model Versions")
    plt.xlabel("Model Version")
    plt.ylabel("Final Win Rate")
    plt.ylim(0, 1)
    plt.show()

def play_test_game(env, agent):
    """
    Play a test game using the trained agent.

    Args:
        env (BoardGameEnv): The game environment.
        agent (QLearningAgent): The trained Q-learning agent.
    """
    state = env.reset()
    done = False
    
    while not done:
        clear_output(wait=True)
        display(env.get_board_image())
        
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)
        state, reward, done, _ = env.step(action)
        
        print(f"AI's move: {action // env.board_size}, {action % env.board_size}")
        
    clear_output(wait=True)
    display(env.get_board_image())
    
    print_game_result(reward)

def print_game_result(reward):
    """Print the result of the game."""
    if reward == 1:
        print("AI wins!")
    elif reward == -1:
        print("AI loses!")
    else:
        print("It's a draw!")