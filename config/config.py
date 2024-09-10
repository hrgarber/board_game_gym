import os

# Environment settings
BOARD_SIZE = 8  # For an 8x8 board

# File paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)

# Logging settings
LOG_LEVEL = "INFO"
