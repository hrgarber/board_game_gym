import unittest
import sys
from .test_board_game import create_suite as create_board_game_suite
from .test_dqn_agent import create_suite as create_dqn_agent_suite
from .test_training import create_suite as create_training_suite


class DetailedTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def run_tests(test_suite, verbosity=2):
    runner = unittest.TextTestRunner(
        verbosity=verbosity, resultclass=DetailedTestResult
    )
    result = runner.run(test_suite)
    return result


def print_summary(result):
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Successes: {len(result.successes)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, error in result.failures:
            print(f"  {test}")
            print(f"    {error}")

    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"  {test}")
            print(f"    {error}")


def run_all_tests():
    all_suites = unittest.TestSuite(
        [
            create_board_game_suite(),
            create_dqn_agent_suite(),
            create_training_suite(),
        ]
    )
    result = run_tests(all_suites)
    print_summary(result)


def run_specific_tests(create_suite_func):
    suite = create_suite_func()
    result = run_tests(suite)
    print_summary(result)


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
            print(
                "Invalid test type. Available options: board_game, dqn_agent, training"
            )
    else:
        run_all_tests()
