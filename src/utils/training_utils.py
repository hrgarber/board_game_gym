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
        list: Episode rewards.
        list: Win rates.
        list: Losses (only for DQN, empty list for Q-Learning).
    """
    episode_rewards = []
    win_rates = []
    losses = []

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
                    losses.append(loss)

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        # Calculate win rate every 100 episodes
        if episode % 100 == 0:
            win_rate = evaluate_agent(env, agent)
            win_rates.append(win_rate)

        if isinstance(agent, DQNAgent) and episode % update_target_every == 0:
            agent.update_target_model()

    return episode_rewards, win_rates, losses
