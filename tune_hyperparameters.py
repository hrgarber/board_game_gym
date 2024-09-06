import argparse
import json
from src.utils.hyperparameter_tuning import grid_search, random_search, bayesian_optimization

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Board Game AI")
    parser.add_argument("agent", choices=["q_learning", "dqn"], help="Type of agent to tune")
    parser.add_argument("method", choices=["grid", "random", "bayesian"], help="Tuning method to use")
    parser.add_argument("--config", default="tuning_config.json", help="Path to tuning configuration file")
    parser.add_argument("--output", default="tuning_results.json", help="Path to save tuning results")
    args = parser.parse_args()

    # Load tuning configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Perform tuning based on the selected method
    if args.method == "grid":
        results = grid_search(args.agent, config[args.agent]["param_grid"])
    elif args.method == "random":
        results = random_search(args.agent, config[args.agent]["param_ranges"])
    elif args.method == "bayesian":
        results = bayesian_optimization(args.agent, config[args.agent]["param_ranges"])

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Tuning completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
import argparse
import json
from src.utils.hyperparameter_tuning import grid_search, random_search, bayesian_optimization

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Board Game AI")
    parser.add_argument("agent", choices=["q_learning", "dqn"], help="Type of agent to tune")
    parser.add_argument("method", choices=["grid", "random", "bayesian"], help="Tuning method to use")
    parser.add_argument("--config", default="tuning_config.json", help="Path to tuning configuration file")
    parser.add_argument("--output", default="tuning_results.json", help="Path to save tuning results")
    args = parser.parse_args()

    # Load tuning configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Perform tuning based on the selected method
    if args.method == "grid":
        results = grid_search(args.agent, config[args.agent]["param_grid"])
    elif args.method == "random":
        results = random_search(args.agent, config[args.agent]["param_ranges"])
    elif args.method == "bayesian":
        results = bayesian_optimization(args.agent, config[args.agent]["param_ranges"])

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Tuning completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
