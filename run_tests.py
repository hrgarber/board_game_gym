import os
import sys
import pytest

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add the tests directory to the Python path
tests_dir = os.path.join(project_root, "tests")
sys.path.insert(0, tests_dir)

if __name__ == "__main__":
    # Run pytest
    pytest.main(["-v", "tests/"])