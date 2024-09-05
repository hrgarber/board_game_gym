import unittest
import numpy as np
import torch
from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent

class TestCase(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_q_learning_agent(self):
        return QLearningAgent(self.state_size, self.action_size)
    
    def create_dqn_agent(self):
        return DQNAgent(self.state_size, self.action_size, self.device)
    
    def assert_valid_action(self, action):
        self.assertIsInstance(action, (int, np.int64))
        self.assertTrue(0 <= action < self.action_size)
    
    def assert_valid_state(self, state):
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (self.state_size,))
    
    def assert_valid_reward(self, reward):
        self.assertIsInstance(reward, (int, float, np.float32, np.float64))
    
    def assert_valid_done(self, done):
        self.assertIsInstance(done, (bool, np.bool_))
