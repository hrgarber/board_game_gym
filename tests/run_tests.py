import unittest
from .test_board_game import create_suite as create_board_game_suite
from .test_dqn_agent import create_suite as create_dqn_agent_suite
from .test_training import create_suite as create_training_suite

class TestResult(unittest.TestResult):
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.failed_tests = []

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.failed_tests.append(test)

    def addError(self, test, err):
        super().addError(test, err)
        self.failed_tests.append(test)

def run_tests(test_suite):
    runner = unittest.TextTestRunner(verbosity=0, resultclass=TestResult, stream=None)
    result = runner.run(test_suite)
    return result.failed_tests

def run_all_tests():
    all_suites = [
        create_board_game_suite(),
        create_dqn_agent_suite(),
        create_training_suite()
    ]
    total_tests = 0
    failed_tests = []

    for suite in all_suites:
        total_tests += suite.countTestCases()
        failed_tests.extend(run_tests(suite))

    print_summary(total_tests, failed_tests)

def run_specific_tests(create_suite_func):
    suite = create_suite_func()
    total_tests = suite.countTestCases()
    failed_tests = run_tests(suite)

    print_summary(total_tests, failed_tests)

def print_summary(total_tests, failed_tests):
    print(f"\nRan {total_tests} tests")
    if failed_tests:
        print(f"FAILED (failures={len(failed_tests)})")
        for test in failed_tests:
            print(f"  {test}")
    else:
        print("OK")

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
