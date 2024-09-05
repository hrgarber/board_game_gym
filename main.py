import numpy as np
from src.environments.board_game_env import BoardGameEnv
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent

def play_game(env, agent):
    """
    Play a game against the trained AI agent.

    Args:
        env (BoardGameEnv): The game environment.
        agent (QLearningAgent): The trained AI agent.
    """
    state = env.reset()
    done = False
    env.render()
    
    while not done:
        if env.current_player == 1:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)
            print(f"AI chooses action: {action // env.board_size}, {action % env.board_size}")
        else:
            while True:
                try:
                    row = int(input("Enter row (0-7): "))
                    col = int(input("Enter column (0-7): "))
                    action = row * env.board_size + col
                    if action in env.get_valid_actions():
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input. Please enter numbers.")
        
        state, reward, done, _ = env.step(action)
        env.render()
    
    if reward == 1:
        print("AI wins!")
    elif reward == -1:
        print("You win!")
    else:
        print("It's a draw!")

def main():
    """
    Main function to run the game and interact with the user.
    """
    env = BoardGameEnv()
    agent = QLearningAgent(state_size=env.board_size * env.board_size, action_size=env.board_size * env.board_size)

    while True:
        print("\n1. Play game")
        print("2. Load trained AI")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            play_game(env, agent)
        elif choice == "2":
            filename = input("Enter filename to load AI: ")
            agent.load_q_table(filename)
            print(f"AI loaded from {filename}")
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
