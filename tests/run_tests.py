import unittest
from .test_board_game import create_suite as create_board_game_suite
from .test_dqn_agent import create_suite as create_dqn_agent_suite
from .test_training import create_suite as create_training_suite

class TestResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.failed_tests = []

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failed_tests.append(test)

    def addError(self, test, err):
        super().addError(test, err)
        self.failed_tests.append(test)

def run_tests(test_suite):
    result = TestResult()
    runner = unittest.TextTestRunner(verbosity=2, resultclass=TestResult)
    runner.run(test_suite, result=result)
    return result.failed_tests

def run_all_tests():
    all_suites = [
        create_board_game_suite(),
        create_dqn_agent_suite(),
        create_training_suite()
    ]
    failed_tests = []

    for suite in all_suites:
        failed_tests.extend(run_tests(suite))

    while failed_tests:
        print("\nRe-running failed tests:")
        retry_suite = unittest.TestSuite(failed_tests)
        failed_tests = run_tests(retry_suite)

    if not failed_tests:
        print("\nAll tests passed successfully!")

def run_specific_tests(create_suite_func):
    suite = create_suite_func()
    failed_tests = run_tests(suite)

    while failed_tests:
        print("\nRe-running failed tests:")
        retry_suite = unittest.TestSuite(failed_tests)
        failed_tests = run_tests(retry_suite)

    if not failed_tests:
        print("\nAll tests passed successfully!")

if __name__ == "__main__":
    import sys
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
