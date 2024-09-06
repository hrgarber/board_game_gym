import unittest
import torch
from src.environments.board_game_env import BoardGameEnv
from src.agents.dqn_agent import DQNAgent
from src.utils.training_utils import train_agent

class TestNotebookCode(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dqn_agent(self):
        state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        action_size = self.env.action_space.n
        assert state_size == 64, f"Unexpected state_size: {state_size}, should be 64."
        
        agent = DQNAgent(state_size, action_size, self.device)
        state = self.env.reset()
        done = False

        # Ensure the initial state shape matches what is expected
        self.assertEqual(state.shape[0], state_size, f"Unexpected initial state shape: {state.shape}")
        
        # Test agent action shape and forward pass
        action = agent.act(state)
        act_values = agent.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        
        print(f"Act Value Shape: {act_values.shape}")
        
        # Check the shape of act_values matches action_size
        self.assertEqual(act_values.shape[1], action_size, f"Unexpected action values shape: {act_values.shape}")
        
        # Simulate a replay step
        batch_size = 32
        for _ in range(batch_size):
            agent.remember(state, action, 1.0, state, done)

        agent.replay(batch_size)
        
    def test_training_utils(self):
        state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        action_size = self.env.action_space.n
        agent = DQNAgent(state_size, action_size, self.device)
        
        num_episodes = 10
        max_steps = 5
        batch_size = 4
        update_target_every = 5

        dqn_rewards, dqn_win_rates = train_agent(self.env, agent, num_episodes, max_steps, batch_size, update_target_every)
        
        self.assertIsInstance(dqn_rewards, list, "Training rewards should be a list")
        self.assertIsInstance(dqn_win_rates, list, "Win rates should be a list")
    
if __name__ == '__main__':
    unittest.main()

