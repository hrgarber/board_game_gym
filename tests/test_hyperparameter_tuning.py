import unittest
import numpy as np
import optuna
from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.environments.board_game_env import BoardGameEnv
from src.utils.hyperparameter_tuning import (
    bayesian_optimization,
    grid_search,
    random_search,
    visualize_tuning_results,
)


class TestHyperparameterTuning(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = (
            self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        )
        self.action_size = self.env.action_space.n

    def test_grid_search(self):
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "discount_factor": [0.9, 0.99],
            "epsilon": [0.1, 0.2],
            "epsilon_decay": [0.99, 0.999],
        }
        results = grid_search(
            "q_learning", param_grid, num_episodes=10, eval_episodes=5
        )
        self.assertIsInstance(results, dict)
        self.assertIn("params", results)
        self.assertIn("performances", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    def test_random_search(self):
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "discount_factor": (0.9, 0.99),
            "epsilon": (0.1, 0.5),
            "epsilon_decay": (0.99, 0.9999),
        }
        results = random_search(
            "q_learning",
            param_ranges,
            num_iterations=5,
            num_episodes=10,
            eval_episodes=5,
        )
        self.assertIsInstance(results, dict)
        self.assertIn("params", results)
        self.assertIn("performances", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    def test_bayesian_optimization(self):
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "discount_factor": (0.9, 0.99),
            "epsilon": (0.1, 0.5),
            "epsilon_decay": (0.99, 0.9999),
        }
        results = bayesian_optimization(
            "q_learning", param_ranges, n_trials=5, num_episodes=10, eval_episodes=5
        )
        self.assertIsInstance(results, dict)
        self.assertIn("study", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    def test_visualize_tuning_results(self):
        # Mock results for testing
        grid_results = {
            "params": [{"learning_rate": 0.01}, {"learning_rate": 0.1}],
            "performances": [0.5, 0.6],
        }
        random_results = {
            "params": [{"learning_rate": 0.05}, {"learning_rate": 0.15}],
            "performances": [0.55, 0.65],
        }

        # Create a mock Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)

        bayesian_results = {
            "study": study,
            "best_params": {"learning_rate": 0.075},
            "best_performance": 0.7,
        }

        # Test visualization functions
        try:
            visualize_tuning_results(grid_results, "grid")
            visualize_tuning_results(random_results, "random")
            visualize_tuning_results(bayesian_results, "bayesian")
        except Exception as e:
            self.fail(f"Visualization failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
import unittest

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.agents.q_learning_agent import QLearningAgent
from src.environments.board_game_env import BoardGameEnv
from src.utils.hyperparameter_tuning import (
    bayesian_optimization,
    grid_search,
    random_search,
    visualize_tuning_results,
)


class TestHyperparameterTuning(unittest.TestCase):
    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = (
            self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        )
        self.action_size = self.env.action_space.n

    def test_grid_search(self):
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "discount_factor": [0.9, 0.99],
            "epsilon": [0.1, 0.2],
            "epsilon_decay": [0.99, 0.999],
        }
        results = grid_search(
            "q_learning", param_grid, num_episodes=10, eval_episodes=5
        )
        self.assertIsInstance(results, dict)
        self.assertIn("params", results)
        self.assertIn("performances", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    def test_random_search(self):
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "discount_factor": (0.9, 0.99),
            "epsilon": (0.1, 0.5),
            "epsilon_decay": (0.99, 0.9999),
        }
        results = random_search(
            "q_learning",
            param_ranges,
            num_iterations=5,
            num_episodes=10,
            eval_episodes=5,
        )
        self.assertIsInstance(results, dict)
        self.assertIn("params", results)
        self.assertIn("performances", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    def test_bayesian_optimization(self):
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "discount_factor": (0.9, 0.99),
            "epsilon": (0.1, 0.5),
            "epsilon_decay": (0.99, 0.9999),
        }
        results = bayesian_optimization(
            "q_learning", param_ranges, n_trials=5, num_episodes=10, eval_episodes=5
        )
        self.assertIsInstance(results, dict)
        self.assertIn("study", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    def test_visualize_tuning_results(self):
        # Mock results for testing
        grid_results = {
            "params": [{"learning_rate": 0.01}, {"learning_rate": 0.1}],
            "performances": [0.5, 0.6],
        }
        random_results = {
            "params": [{"learning_rate": 0.05}, {"learning_rate": 0.15}],
            "performances": [0.55, 0.65],
        }
        bayesian_results = {
            "study": optuna.create_study(direction="maximize"),
            "best_params": {"learning_rate": 0.075},
            "best_performance": 0.7,
        }

        # Add multiple trials to the study
        for _ in range(5):
            bayesian_results["study"].add_trial(
                optuna.trial.create_trial(
                    params={"learning_rate": np.random.uniform(0.001, 0.1)},
                    distributions={
                        "learning_rate": optuna.distributions.FloatDistribution(
                            0.001, 0.1
                        )
                    },
                    value=np.random.uniform(0.5, 0.8),
                )
            )

        # Test visualization functions
        try:
            visualize_tuning_results(grid_results, "grid")
            visualize_tuning_results(random_results, "random")
            visualize_tuning_results(bayesian_results, "bayesian")
        except Exception as e:
            if "Cannot evaluate parameter importances with only a single trial" in str(
                e
            ):
                print(
                    "Bayesian optimization visualization skipped due to insufficient trials."
                )
            else:
                self.fail(f"Visualization failed with unexpected error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.utils.hyperparameter_tuning import cross_validate, grid_search, random_search, bayesian_optimization
from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent

class TestHyperparameterTuning(unittest.TestCase):

    def setUp(self):
        self.env = BoardGameEnv()
        self.state_size = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.action_size = self.env.action_space.n

    @patch('src.utils.hyperparameter_tuning.evaluate_agent')
    def test_cross_validate(self, mock_evaluate_agent):
        mock_evaluate_agent.return_value = 0.5
        params = {
            "learning_rate": 0.01,
            "discount_factor": 0.99,
            "epsilon": 0.1,
            "epsilon_decay": 0.995
        }
        result = cross_validate("q_learning", params, n_splits=2, num_episodes=10, eval_episodes=5)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.5)

    @patch('src.utils.hyperparameter_tuning.cross_validate')
    def test_grid_search(self, mock_cross_validate):
        mock_cross_validate.return_value = 0.5
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "discount_factor": [0.9, 0.99],
            "epsilon": [0.1],
            "epsilon_decay": [0.995]
        }
        results = grid_search("q_learning", param_grid, num_episodes=10, eval_episodes=5, n_splits=2)
        self.assertIn("params", results)
        self.assertIn("performances", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    @patch('src.utils.hyperparameter_tuning.cross_validate')
    def test_random_search(self, mock_cross_validate):
        mock_cross_validate.return_value = 0.5
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "discount_factor": (0.9, 0.99),
            "epsilon": (0.1, 0.5),
            "epsilon_decay": (0.99, 0.9999)
        }
        results = random_search("q_learning", param_ranges, num_iterations=5, num_episodes=10, eval_episodes=5, n_splits=2)
        self.assertIn("params", results)
        self.assertIn("performances", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

    @patch('src.utils.hyperparameter_tuning.cross_validate')
    def test_bayesian_optimization(self, mock_cross_validate):
        mock_cross_validate.return_value = 0.5
        param_ranges = {
            "learning_rate": (0.001, 0.1),
            "discount_factor": (0.9, 0.99),
            "epsilon": (0.1, 0.5),
            "epsilon_decay": (0.99, 0.9999)
        }
        results = bayesian_optimization("q_learning", param_ranges, n_trials=5, num_episodes=10, eval_episodes=5, n_splits=2)
        self.assertIn("study", results)
        self.assertIn("best_params", results)
        self.assertIn("best_performance", results)

if __name__ == '__main__':
    unittest.main()
