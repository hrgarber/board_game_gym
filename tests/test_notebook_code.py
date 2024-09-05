import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.utils import evaluate_agent, plot_training_results, plot_version_comparison
from src.utils.training_utils import train_agent

class TestNotebookCode(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_agent = QLearningAgent(self.state_size, self.action_size)
        self.dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)
        self.num_episodes = 100
        self.max_steps = 100
        self.batch_size = 32
        self.update_target_every = 10

    def test_environment_initialization(self):
        self.assertIsInstance(self.env, BoardGameEnv)
        self.assertEqual(self.env.observation_space.shape[0], self.state_size)
        self.assertEqual(self.env.action_space.n, self.action_size)

    def test_agent_initialization(self):
        self.assertIsInstance(self.q_agent, QLearningAgent)
        self.assertIsInstance(self.dqn_agent, DQNAgent)

    def test_train_agent_function(self):
        # Test Q-Learning training
        q_rewards, q_win_rates = train_agent(self.env, self.q_agent, self.num_episodes, self.max_steps)
        self.assertEqual(len(q_rewards), self.num_episodes)
        self.assertEqual(len(q_win_rates), self.num_episodes // 100)

        # Test DQN training
        dqn_rewards, dqn_win_rates = train_agent(self.env, self.dqn_agent, self.num_episodes, self.max_steps, self.batch_size, self.update_target_every)
        self.assertEqual(len(dqn_rewards), self.num_episodes)
        self.assertEqual(len(dqn_win_rates), self.num_episodes // 100)

    def test_evaluate_agent(self):
        win_rate = evaluate_agent(self.env, self.q_agent, num_episodes=10)
        self.assertIsInstance(win_rate, float)
        self.assertTrue(0 <= win_rate <= 1)

    def test_plot_training_results(self):
        rewards = [1, 2, 3, 4, 5]
        win_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock plt.show to avoid displaying the plot during testing
        plt.show = lambda: None

        try:
            plot_training_results(rewards, win_rates, "Test Agent")
        except Exception as e:
            self.fail(f"plot_training_results raised an exception: {e}")

    def test_plot_version_comparison(self):
        version1_results = [1, 2, 3, 4, 5]
        version2_results = [2, 3, 4, 5, 6]
        
        # Mock plt.show to avoid displaying the plot during testing
        plt.show = lambda: None

        try:
            plot_version_comparison(self.env, 'models')
        except Exception as e:
            self.fail(f"plot_version_comparison raised an exception: {e}")

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
        os.remove('test_q_model.json')
        os.remove('test_dqn_model.pth')

    def test_compare_agents(self):
        def compare_agents(env, dqn_agent, num_games=10):
            win_rate = evaluate_agent(env, dqn_agent, num_games)
            self.assertIsInstance(win_rate, float)
            self.assertTrue(0 <= win_rate <= 1)

        compare_agents(self.env, self.dqn_agent)

    def test_play_test_game(self):
        def play_test_game(env, agent, agent_name):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

            self.assertIsInstance(total_reward, (int, float))

        play_test_game(self.env, self.dqn_agent, "DQN")

if __name__ == '__main__':
    unittest.main()

