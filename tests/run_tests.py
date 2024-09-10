import logging
import os
import sys
import unittest

# Add the parent directory to sys.path to allow importing from sibling directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_board_game import create_suite as create_board_game_suite
from tests.test_utils import TestCase as UtilsTestCase

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Set to WARNING to suppress debug messages


class DetailedTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def run_tests(test_suite, verbosity=0):
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        resultclass=DetailedTestResult,
        stream=open(os.devnull, "w"),
    )
    result = runner.run(test_suite)
    return result


def print_summary(result):
    print("\nTest Summary:")
    print(f"Ran {result.testsRun} tests")
    print(f"Successes: {len(result.successes)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\nAll tests passed successfully!")
    else:
        print("\nFailed tests:")
        for test, error in result.failures + result.errors:
            print(f"  {test}")
            print(f"    {error}\n")


def run_all_tests():
    all_suites = unittest.TestSuite([
        create_board_game_suite(),
        unittest.TestLoader().loadTestsFromTestCase(UtilsTestCase)
    ])
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
        elif test_type == "utils":
            run_specific_tests(lambda: unittest.TestLoader().loadTestsFromTestCase(UtilsTestCase))
        else:
            print("Invalid test type. Available options: board_game, utils")
    else:
        run_all_tests()
