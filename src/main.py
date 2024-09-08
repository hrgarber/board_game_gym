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
        default="tuning_config.json",
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
        default="tuning_config.json",
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
