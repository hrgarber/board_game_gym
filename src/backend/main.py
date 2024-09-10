import argparse
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from environments.board_game_env import BoardGameEnv

def main():
    parser = argparse.ArgumentParser(description="Board Game Gym Environment")
    parser.add_argument("--board_size", type=int, default=8, help="Size of the board")
    args = parser.parse_args()

    env = BoardGameEnv(board_size=args.board_size)
    print("Board Game Gym Environment initialized.")

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        env.render()
        valid_actions = env.get_valid_actions()
        
        if not valid_actions:
            print("No valid actions. Game over!")
            break

        print("Valid actions:", valid_actions)
        
        while True:
            try:
                action = int(input("Enter your action (0-63): "))
                if action in valid_actions:
                    break
                else:
                    print("Invalid action. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        state, reward, done, _ = env.step(action)
        total_reward += reward

        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}")

    env.render()
    print("Game Over!")
    print(f"Final Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
