import os
from config.config import IMAGE_DIR

def save_tuning_results(fig, filename):
    """
    Save the tuning results figure to the specified image directory.
    
    Args:
    fig (matplotlib.figure.Figure): The figure to save
    filename (str): The name of the file to save
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)
    full_path = os.path.join(IMAGE_DIR, filename)
    fig.savefig(full_path)
    print(f"Saved figure to {full_path}")

# Example usage:
# save_tuning_results(fig, "grid_tuning_results.png")
# save_tuning_results(fig, "random_tuning_results.png")
