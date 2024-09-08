import os
import sys
import unittest

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.test_board_game import create_suite as create_board_game_suite
from tests.test_dqn_agent import create_suite as create_dqn_agent_suite
from tests.test_training import create_suite as create_training_suite

def run_all_tests():
    test_suite = unittest.TestSuite([
        create_board_game_suite(),
        create_dqn_agent_suite(),
        create_training_suite()
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

def run_board_game_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_board_game_suite())

def run_dqn_agent_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_dqn_agent_suite())

def run_training_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_training_suite())

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "board_game":
            run_board_game_tests()
        elif test_type == "dqn_agent":
            run_dqn_agent_tests()
        elif test_type == "training":
            run_training_tests()
        else:
            print("Invalid test type. Available options: board_game, dqn_agent, training")
    else:
        run_all_tests()
