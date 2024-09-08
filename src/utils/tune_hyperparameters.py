import argparse
import json
import os
import sys
from typing import Any, Dict

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.hyperparameter_tuning import (
    bayesian_optimization,
    grid_search,
    random_search,
    visualize_tuning_results,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the tuning configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save the tuning results to a JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Board Game AI"
    )
    parser.add_argument(
        "agent", choices=["q_learning", "dqn"], help="Type of agent to tune"
    )
    parser.add_argument(
        "method", choices=["grid", "random", "bayesian"], help="Tuning method to use"
    )
    parser.add_argument(
        "--config",
        default="config/tuning_config.json",
        help="Path to tuning configuration file",
    )
    parser.add_argument(
        "--output", default="tuning_results.json", help="Path to save tuning results"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    tuning_methods = {
        "grid": grid_search,
        "random": random_search,
        "bayesian": bayesian_optimization,
    }

    results = tuning_methods[args.method](
        args.agent,
        config[args.agent]["param_grid" if args.method == "grid" else "param_ranges"],
    )

    output_dir = os.path.join(project_root, "output", "hyperparameter_tuning")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, args.output)
    save_results(results, output_file)
    print(f"Tuning completed. Results saved to {output_file}")

    # Visualize the results
    visualize_tuning_results(results, args.method)
    print(f"Visualization saved to {output_dir}/{args.method}_tuning_results.png")


if __name__ == "__main__":
    main()
