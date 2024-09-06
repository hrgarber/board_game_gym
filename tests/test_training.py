from tests.test_utils import TestCase
import numpy as np
from src.utils.utils import evaluate_agent, plot_training_results, compare_agents, plot_agent_comparison
from src.utils.training_utils import train_agent
import matplotlib.pyplot as plt

class TestTraining(TestCase):
    def setUp(self):
        super().setUp()
        self.q_learning_agent = self.create_q_learning_agent()
        self.dqn_agent = self.create_dqn_agent()

    def test_q_learning_training_step(self):
        self._test_training_step(self.q_learning_agent)

    def test_dqn_training_step(self):
        self._test_training_step(self.dqn_agent)

    def _test_training_step(self, agent):
        state = self.env.reset()
        valid_actions = self.env.get_valid_actions()
        action = agent.act(state)
        next_state, reward, done, _ = self.env.step(action)

        self.assert_valid_state(state)
        self.assert_valid_state(next_state)
        self.assert_valid_reward(reward)
        self.assert_valid_done(done)
        self.assert_valid_action(action)
        self.assertIn(action, valid_actions)

    def test_q_learning_episode_completion(self):
        self._test_episode_completion(self.q_learning_agent)

    def test_dqn_episode_completion(self):
        self._test_episode_completion(self.dqn_agent)

    def _test_episode_completion(self, agent):
        episode_reward = 0
        state = self.env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = self.env.step(action)
            
            if isinstance(agent, type(self.q_learning_agent)):
                agent.update_q_value(state, action, reward, next_state)
            else:
                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) > agent.batch_size:
                    agent.replay(agent.batch_size)
            
            state = next_state
            episode_reward += reward
        
        self.assert_valid_reward(episode_reward)
        self.assertTrue(done)

    def test_win_rate_calculation(self):
        num_episodes = 100
        rewards = [1] * 50 + [-1] * 50  # Simulate 50 wins and 50 losses
        win_rate = sum(rewards) / (2 * num_episodes) + 0.5
        
        self.assertAlmostEqual(win_rate, 0.5, places=2)

    def test_evaluate_agent_q_learning(self):
        win_rate = evaluate_agent(self.env, self.q_learning_agent, num_episodes=100)
        self.assertIsInstance(win_rate, float)
        self.assertTrue(0 <= win_rate <= 1)

    def test_evaluate_agent_dqn(self):
        win_rate = evaluate_agent(self.env, self.dqn_agent, num_episodes=100)
        self.assertIsInstance(win_rate, float)
        self.assertTrue(0 <= win_rate <= 1)

    def test_plot_training_results(self):
        rewards = [1, 2, 3, 4, 5]
        win_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
        agent_name = "Test Agent"

        # Mock plt.show to avoid displaying the plot during testing
        plt.show = lambda: None

        # Call the function and check if it runs without errors
        try:
            plot_training_results(rewards, win_rates, agent_name)
        except Exception as e:
            self.fail(f"plot_training_results raised an exception: {e}")

    def test_compare_agents(self):
        q_win_rate, dqn_win_rate = compare_agents(self.env, self.q_learning_agent, self.dqn_agent, num_episodes=10)

        self.assertIsInstance(q_win_rate, float)
        self.assertIsInstance(dqn_win_rate, float)
        self.assertTrue(0 <= q_win_rate <= 1)
        self.assertTrue(0 <= dqn_win_rate <= 1)

    def test_plot_agent_comparison(self):
        q_win_rate, dqn_win_rate = 0.6, 0.7
        
        # Mock plt.show to avoid displaying the plot during testing
        plt.show = lambda: None

        try:
            plot_agent_comparison(q_win_rate, dqn_win_rate)
        except Exception as e:
            self.fail(f"plot_agent_comparison raised an exception: {e}")

    def test_train_agent(self):
        num_episodes = 10
        max_steps = 100
        batch_size = 32
        update_target_every = 5

        # Test Q-Learning agent training
        q_results = train_agent(self.env, self.q_learning_agent, num_episodes, max_steps)
        self.assertIsInstance(q_results, tuple)
        self.assertEqual(len(q_results), 2)
        self.assertEqual(len(q_results[0]), num_episodes)
        self.assertEqual(len(q_results[1]), num_episodes // 100 + 1)

        # Test DQN agent training
        dqn_results = train_agent(self.env, self.dqn_agent, num_episodes, max_steps, batch_size, update_target_every)
        self.assertIsInstance(dqn_results, tuple)
        self.assertEqual(len(dqn_results), 2)
        self.assertEqual(len(dqn_results[0]), num_episodes)
        self.assertEqual(len(dqn_results[1]), num_episodes // 100 + 1)

if __name__ == "__main__":
    from unittest import main
    main()

