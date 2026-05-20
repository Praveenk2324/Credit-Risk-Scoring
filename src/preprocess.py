import pandas as pd
import os

def main():
    os.makedirs('data/processed', exist_ok=True)

    print("Preprocessing data...")

    df = pd.read_csv('data/raw/current_data.csv', index_col=0)

    df.to_parquet('data/processed/train.parquet')
    print("Saved processed data to data/processed/train.parquet")

if __name__ == "__main__":
    main()