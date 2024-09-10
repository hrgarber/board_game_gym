import os
import sys
import unittest

# Add the project root to sys.path to allow importing from sibling directories
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from src.tests.test_board_game import create_suite as create_board_game_suite
from src.tests.test_utils import TestCase as UtilsTestCase
from src.backend.logger import logger

class DetailedTestResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)
        logger.info(f"Test passed: {test}")

    def addError(self, test, err):
        super().addError(test, err)
        logger.error(f"Test error: {test}\n{err}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        logger.error(f"Test failed: {test}\n{err}")

def run_tests(test_suite, verbosity=2):
    logger.info("Starting test run")
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        resultclass=DetailedTestResult,
    )
    result = runner.run(test_suite)
    return result

def print_summary(result):
    logger.info("\nTest Summary:")
    logger.info(f"Ran {result.testsRun} tests")
    logger.info(f"Successes: {len(result.successes)}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        logger.info("\nAll tests passed successfully!")
    else:
        logger.error("\nFailed tests:")
        for test, error in result.failures + result.errors:
            logger.error(f"  {test}")
            logger.error(f"    {error}\n")

def run_all_tests():
    logger.info("Running all tests")
    all_suites = unittest.TestSuite([
        create_board_game_suite(),
        unittest.TestLoader().loadTestsFromTestCase(UtilsTestCase)
    ])
    result = run_tests(all_suites)
    print_summary(result)

def run_specific_tests(create_suite_func):
    logger.info(f"Running specific test suite: {create_suite_func.__name__}")
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
            logger.error(f"Invalid test type: {test_type}. Available options: board_game, utils")
    else:
        run_all_tests()
