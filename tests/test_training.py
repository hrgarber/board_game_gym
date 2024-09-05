from tests.test_utils import TestCase
import numpy as np
from src.utils.utils import evaluate_agent

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

if __name__ == "__main__":
    from unittest import main
    main()
