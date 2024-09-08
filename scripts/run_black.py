import os
import subprocess
import sys

def run_black(directory):
    """Run Black on all Python files in the given directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Running Black on {file_path}")
                subprocess.run(['black', file_path])

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(project_root, 'src')
    tests_dir = os.path.join(project_root, 'tests')

    run_black(src_dir)
    run_black(tests_dir)

    print("Black formatting complete.")
