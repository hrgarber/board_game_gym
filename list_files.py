import os

def list_first_layer():
    for root, dirs, files in os.walk('.', topdown=True):
        print("Directories:")
        for name in dirs:
            print(f"  {name}")
        print("\nFiles:")
        for name in files:
            print(f"  {name}")
        break  # Only process the first layer

if __name__ == "__main__":
    list_first_layer()
