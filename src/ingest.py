import shutil
import os

def main():
    os.makedirs('data/raw', exist_ok=True)

    source = 'data/simulation_stream/01_base_data.csv'
    destination = 'data/raw/current_data.csv'

    print("Ingestion data...")
    shutil.copy(source, destination)
    print("Saved to data/raw/current_data.csv")

if __name__ == '__main__':
    main()