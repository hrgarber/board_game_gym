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

# Training settings
NUM_EPISODES = 1000

# File paths
MODEL_DIR = "models"
IMAGE_DIR = "output/images"
