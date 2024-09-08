import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tests.run_tests import run_all_tests, run_specific_tests
from tests.test_board_game import create_suite as create_board_game_suite
from tests.test_dqn_agent import create_suite as create_dqn_agent_suite
from tests.test_training import create_suite as create_training_suite

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        if test_type == "board_game":
            run_specific_tests(create_board_game_suite)
        elif test_type == "dqn_agent":
            run_specific_tests(create_dqn_agent_suite)
        elif test_type == "training":
            run_specific_tests(create_training_suite)
        else:
            print("Invalid test type. Available options: board_game, dqn_agent, training")
    else:
        run_all_tests()
