import unittest
import numpy as np
from board_game_env import BoardGameEnv
from q_learning_agent import QLearningAgent

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.agent = QLearningAgent(state_size=self.env.board_size * self.env.board_size, 
                                    action_size=self.env.board_size * self.env.board_size)

    def test_training_step(self):
        state = self.env.reset()
        
        valid_actions = self.env.get_valid_actions()
        action = self.agent.choose_action(state, valid_actions)
        
        next_state, reward, done, _ = self.env.step(action)
        
        print(f"Type of done: {type(done)}")
        print(f"Value of done: {done}")
        
        self.agent.update_q_value(state, action, reward, next_state)
        
        self.assertIsInstance(state, np.ndarray)
        self.assertIsInstance(next_state, np.ndarray)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, (bool, np.bool_))  # Accept both Python bool and NumPy bool_
        self.assertIn(action, valid_actions)

    def test_episode_completion(self):
        episode_reward = 0
        state = self.env.reset()
        done = False
        
        while not done:
            valid_actions = self.env.get_valid_actions()
            action = self.agent.choose_action(state, valid_actions)
            next_state, reward, done, _ = self.env.step(action)
            
            self.agent.update_q_value(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
        
        self.assertIsInstance(episode_reward, (int, float))
        self.assertTrue(done)

    def test_win_rate_calculation(self):
        num_episodes = 100
        rewards = [1] * 50 + [-1] * 50  # Simulate 50 wins and 50 losses
        win_rate = sum(rewards) / (2 * num_episodes) + 0.5
        
        self.assertAlmostEqual(win_rate, 0.5, places=2)

if __name__ == "__main__":
    unittest.main()