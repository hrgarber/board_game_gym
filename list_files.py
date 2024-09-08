import os

def list_first_layer():
    ignore_files = {'pytest.ini', 'config.py', 'list_files.py'}
    for root, dirs, files in os.walk('.', topdown=True):
        print("Directories:")
        for name in dirs:
            if not name.startswith('.'):  # Ignore hidden directories
                print(f"  {name}")
        print("\nFiles:")
        for name in files:
            if name not in ignore_files and not name.startswith('.'):  # Ignore specific files and hidden files
                print(f"  {name}")
        break  # Only process the first layer

if __name__ == "__main__":
    list_first_layer()
