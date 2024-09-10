import os

# General settings
DEVICE = "cpu"  # We'll update this when we implement the new RL approach

# Agent settings
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 0.1

# Environment settings
MAX_STEPS = 1000
BOARD_SIZE = 3  # For a 3x3 board

# Training settings
NUM_EPISODES = 1000
SAVE_INTERVAL = 100  # Save model every 100 episodes

# Evaluation settings
EVAL_EPISODES = 100

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for directory in [MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging settings
LOG_LEVEL = "INFO"
