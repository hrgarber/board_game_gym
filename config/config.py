import os

import torch

# General settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Agent settings
LEARNING_RATE = 1e-3
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
UPDATE_TARGET_EVERY = 100

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
IMAGE_DIR = os.path.join(BASE_DIR, "output", "images")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
for directory in [MODEL_DIR, IMAGE_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Logging settings
LOG_LEVEL = "INFO"
