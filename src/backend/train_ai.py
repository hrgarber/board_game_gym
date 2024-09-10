import gym
import numpy as np
import os
import subprocess
import argparse
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environments.board_game_env import BoardGameEnv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom callback for TensorBoard logging
class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.step = 0

    def _on_step(self) -> bool:
        self.step += 1
        if self.step % 1000 == 0:  # Log every 1000 steps
            self.logger.record('custom/reward_mean', np.mean(self.locals['rewards']))
            logging.info(f"Step {self.step}: Mean reward = {np.mean(self.locals['rewards']):.2f}")
        return True

def train_ai(total_timesteps=100000, learning_rate=0.0003, n_steps=2048):
    logging.info(f"Starting AI training with {total_timesteps} timesteps, learning rate {learning_rate}, and {n_steps} steps per update")
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: BoardGameEnv()])

    # Initialize the model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/",
                learning_rate=learning_rate, n_steps=n_steps)

    # Create callback
    callback = TensorBoardCallback()

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the model
    model.save("ppo_board_game")
    logging.info("Model training completed and saved as 'ppo_board_game'")

    return model, env

def evaluate_ai(model, env, n_eval_episodes=10):
    logging.info(f"Evaluating AI model over {n_eval_episodes} episodes")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    logging.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

def test_ai(model, env, n_test_episodes=5):
    logging.info(f"Testing AI model over {n_test_episodes} episodes")
    for episode in range(n_test_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        logging.info(f"Episode {episode + 1} total reward: {total_reward}")

def start_tensorboard():
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir=./tensorboard_logs"])
    logging.info("TensorBoard started. Open http://localhost:6006/ in your browser to view.")
    return tensorboard_process

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an AI for the Board Game Gym")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--test_episodes", type=int, default=5, help="Number of episodes for testing")
    args = parser.parse_args()

    logging.info("Starting AI training process")
    model, env = train_ai(total_timesteps=args.timesteps, learning_rate=args.lr, n_steps=args.n_steps)
    
    logging.info("Evaluating trained model")
    evaluate_ai(model, env, n_eval_episodes=args.eval_episodes)
    
    logging.info("Testing trained model")
    test_ai(model, env, n_test_episodes=args.test_episodes)
    
    logging.info("Starting TensorBoard")
    tensorboard_process = start_tensorboard()
    
    input("Press Enter to exit and close TensorBoard...")
    tensorboard_process.terminate()

if __name__ == "__main__":
    main()