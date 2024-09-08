import os

def list_first_layer():
    ignore_files = {
        'list_files.py', 'README.md',
        'hyperparameter_config.json', 'error_log.txt'
    }
    ignore_extensions = {'.pyc', '.log', '.png', '.txt'}
    
    for root, dirs, files in os.walk('.', topdown=True):
        print("Directories:")
        for name in sorted(dirs):
            if not name.startswith('.') and name != '__pycache__':  # Ignore hidden directories and __pycache__
                print(f"  {name}")
        
        print("\nFiles:")
        for name in sorted(files):
            if (name not in ignore_files and
                not name.startswith('.') and
                not any(name.endswith(ext) for ext in ignore_extensions)):
                print(f"  {name}")
        break  # Only process the first layer

if __name__ == "__main__":
    list_first_layer()
