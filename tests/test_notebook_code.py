import unittest
import torch
from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.training_utils import train_agent
from src.utils.utils import evaluate_agent, plot_training_results, compare_agents, plot_agent_comparison

class TestNotebookCode(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.action_size = self.env.action_space.n

    def test_q_learning_agent(self):
        agent = QLearningAgent(self.state_size, self.action_size)
        state = self.env.reset()
        action = agent.act(state)
        
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)

    def test_dqn_agent(self):
        agent = DQNAgent(self.state_size, self.action_size, self.device)
        state = self.env.reset()
        action = agent.act(state)
        
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_size)

        # Test agent action shape and forward pass
        act_values = agent.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        self.assertEqual(act_values.shape[1], self.action_size)

    def test_training_utils(self):
        q_agent = QLearningAgent(self.state_size, self.action_size)
        dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)
        
        num_episodes = 10
        max_steps = 5
        batch_size = 4
        update_target_every = 5

        q_rewards, q_win_rates = train_agent(self.env, q_agent, num_episodes, max_steps)
        dqn_rewards, dqn_win_rates = train_agent(self.env, dqn_agent, num_episodes, max_steps, batch_size, update_target_every)
        
        self.assertIsInstance(q_rewards, list)
        self.assertIsInstance(q_win_rates, list)
        self.assertIsInstance(dqn_rewards, list)
        self.assertIsInstance(dqn_win_rates, list)

    def test_evaluate_agent(self):
        q_agent = QLearningAgent(self.state_size, self.action_size)
        dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)

        q_win_rate = evaluate_agent(self.env, q_agent, num_episodes=10)
        dqn_win_rate = evaluate_agent(self.env, dqn_agent, num_episodes=10)

        self.assertIsInstance(q_win_rate, float)
        self.assertIsInstance(dqn_win_rate, float)
        self.assertTrue(0 <= q_win_rate <= 1)
        self.assertTrue(0 <= dqn_win_rate <= 1)

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

    def test_compare_agents(self):
        q_agent = QLearningAgent(self.state_size, self.action_size)
        dqn_agent = DQNAgent(self.state_size, self.action_size, self.device)

        q_win_rate, dqn_win_rate = compare_agents(self.env, q_agent, dqn_agent, num_episodes=10)

        self.assertIsInstance(q_win_rate, float)
        self.assertIsInstance(dqn_win_rate, float)
        self.assertTrue(0 <= q_win_rate <= 1)
        self.assertTrue(0 <= dqn_win_rate <= 1)

    def test_plot_agent_comparison(self):
        # Mock plt.show to avoid displaying the plot during testing
        import matplotlib.pyplot as plt
        plt.show = lambda: None

        try:
            plot_agent_comparison(0.6, 0.7)
        except Exception as e:
            self.fail(f"plot_agent_comparison raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()

