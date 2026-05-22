import shutil
import os
import sys

def main():
    os.makedirs('data/raw', exist_ok=True)

    batch_file = sys.argv[1] if len(sys.argv) > 1 else '01_base_data.csv'

    source = f'data/simulation_stream/{batch_file}'
    destination = 'data/raw/current_data.csv'

    print("Ingestion data...")
    shutil.copy(source, destination)
    print("Saved to data/raw/current_data.csv")

if __name__ == '__main__':
    main()