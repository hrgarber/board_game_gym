import unittest
import numpy as np
import torch
from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.utils import evaluate_agent, plot_training_results

class TestNotebookCode(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_agent = QLearningAgent(self.state_size, self.action_size)
        self.dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)

    def test_environment_initialization(self):
        self.assertIsInstance(self.env, BoardGameEnv)
        self.assertEqual(self.env.board_size, 8)

    def test_agent_initialization(self):
        self.assertIsInstance(self.q_agent, QLearningAgent)
        self.assertIsInstance(self.dqn_agent, DQNAgent)

    def test_training_function(self):
        def train_agent(env, agent, num_episodes, max_steps, batch_size=None, update_target_every=None):
            episode_rewards = []
            win_rates = []

            for episode in range(num_episodes):
                state = env.reset()
                total_reward = 0
                for step in range(max_steps):
                    action = agent.act(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    if isinstance(agent, QLearningAgent):
                        agent.update_q_value(state, action, reward, next_state)
                    else:  # DQNAgent
                        agent.remember(state, action, reward, next_state, done)
                        if len(agent.memory) > batch_size:
                            agent.replay(batch_size)
                    
                    state = next_state
                    total_reward += reward

                    if done:
                        break

                episode_rewards.append(total_reward)
                
                if episode % 100 == 0:
                    win_rate = evaluate_agent(env, agent)
                    win_rates.append(win_rate)

                if isinstance(agent, DQNAgent) and episode % update_target_every == 0:
                    agent.update_target_model()

            return episode_rewards, win_rates

        # Test Q-Learning training
        q_rewards, q_win_rates = train_agent(self.env, self.q_agent, num_episodes=100, max_steps=100)
        self.assertEqual(len(q_rewards), 100)
        self.assertEqual(len(q_win_rates), 1)

        # Test DQN training
        dqn_rewards, dqn_win_rates = train_agent(self.env, self.dqn_agent, num_episodes=100, max_steps=100, batch_size=32, update_target_every=10)
        self.assertEqual(len(dqn_rewards), 100)
        self.assertEqual(len(dqn_win_rates), 1)

    def test_evaluate_agent(self):
        win_rate = evaluate_agent(self.env, self.q_agent, num_episodes=10)
        self.assertIsInstance(win_rate, float)
        self.assertTrue(0 <= win_rate <= 1)

    def test_plot_training_results(self):
        rewards = [1, 2, 3, 4, 5]
        win_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock plt.show to avoid displaying the plot during testing
        import matplotlib.pyplot as plt
        plt.show = lambda: None

        try:
            plot_training_results(rewards, win_rates, "Test Agent")
        except Exception as e:
            self.fail(f"plot_training_results raised an exception: {e}")

    def test_model_saving_and_loading(self):
        # Test Q-Learning model saving and loading
        self.q_agent.save_model('test_q_model.json')
        new_q_agent = QLearningAgent(self.state_size, self.action_size)
        new_q_agent.load_model('test_q_model.json')
        self.assertEqual(self.q_agent.q_table, new_q_agent.q_table)

        # Test DQN model saving and loading
        self.dqn_agent.save('test_dqn_model.pth')
        new_dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)
        new_dqn_agent.load('test_dqn_model.pth')
        for param1, param2 in zip(self.dqn_agent.model.parameters(), new_dqn_agent.model.parameters()):
            self.assertTrue(torch.equal(param1, param2))

        # Clean up test files
        import os
        os.remove('test_q_model.json')
        os.remove('test_dqn_model.pth')

if __name__ == '__main__':
    unittest.main()
