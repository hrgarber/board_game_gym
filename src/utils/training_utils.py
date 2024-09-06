import numpy as np
import torch
from tqdm import tqdm
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.utils import evaluate_agent

def train_agent(env, agent, num_episodes, max_steps, batch_size=None, update_target_every=None):
    """Train the agent (Q-Learning or DQN).

    Args:
        env (BoardGameEnv): The game environment.
        agent (QLearningAgent or DQNAgent): The agent to train.
        num_episodes (int): Number of episodes to train.
        max_steps (int): Maximum steps per episode.
        batch_size (int): Batch size for DQN training (ignored for Q-Learning).
        update_target_every (int): Number of episodes between target network updates for DQN (ignored for Q-Learning).

    Returns:
        tuple: A tuple containing two lists (rewards, win_rates).
    """
    rewards = []
    win_rates = []

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        assert state.shape[0] == 64, f"Unexpected state shape: {state.shape[0]}, should be 64."
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            if isinstance(agent, QLearningAgent):
                agent.update_q_value(state, action, reward, next_state)
            else:  # DQNAgent
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > batch_size:
                    loss = agent.replay(batch_size)
                    print(f"Episode {episode}, Step {step}, Loss: {loss}")

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

        # Calculate win rate every 100 episodes
        if episode % 100 == 0:
            win_rate = evaluate_agent(env, agent)
            win_rates.append(win_rate)
            print(f"Episode {episode}, Win Rate: {win_rate}, Total Reward: {total_reward}")

        if isinstance(agent, DQNAgent) and episode % update_target_every == 0:
            agent.update_target_model()
            print(f"Updated target model at episode {episode}")

        if isinstance(agent, QLearningAgent):
            agent.decay_epsilon()
        elif isinstance(agent, DQNAgent):
            old_epsilon = agent.epsilon
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            print(f"Epsilon decayed from {old_epsilon} to {agent.epsilon}")

    return rewards, win_rates
