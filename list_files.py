import os

def list_first_layer():
    ignore_dirs = {
        '__pycache__',
        'output'
    }
    
    ignore_files = {
        'list_files.py',
        'error_log.txt',
        'errors.txt',
        'hyperparameter_tuning_20240906_001503.log',
        'pylint-report.txt'
    }
    
    for root, dirs, files in os.walk('.', topdown=True):
        print("Directories:")
        for name in sorted(dirs):
            if not name.startswith('.') and name not in ignore_dirs:
                print(f"  {name}")
        
        print("\nFiles:")
        for name in sorted(files):
            if name not in ignore_files and not name.startswith('.'):
                print(f"  {name}")
        break  # Only process the first layer

if __name__ == "__main__":
    list_first_layer()
