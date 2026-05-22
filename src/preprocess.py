import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clean_and_scale(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    X = df.drop('SeriousDlqin2yrs', axis=1).copy()
    y = df['SeriousDlqin2yrs'].copy()

    X['MonthlyIncome_is_missing'] = np.where(X['MonthlyIncome'].isnull(), 1, 0)
    income_imputer = SimpleImputer(strategy='median')
    X['MonthlyIncome'] = income_imputer.fit_transform(X[['MonthlyIncome']])

    dep_imputer = SimpleImputer(strategy='most_frequent')
    X['NumberOfDependents'] = dep_imputer.fit_transform(X[['NumberOfDependents']])

    median_age = X[X['age'] > 0]['age'].median()
    X.loc[X['age'] == 0, 'age'] = median_age

    late_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse', 
        'NumberOfTime60-89DaysPastDueNotWorse', 
        'NumberOfTimes90DaysLate'
    ]
    for col in late_cols:
        X.loc[X[col].isin([96, 98]), col] = 0

    X['RevolvingUtilizationOfUnsecuredLines'] = X['RevolvingUtilizationOfUnsecuredLines'].clip(upper=2.0)
    debt_cap = X['DebtRatio'].quantile(0.95)
    X['DebtRatio'] = X['DebtRatio'].clip(upper=debt_cap)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    joblib.dump(scaler, 'data/processed/scaler.pkl')

    return X_train_scaled, y

def main():
    os.makedirs('data/preprocessed', exist_ok=True)
    print("Processing data...")

    df = pd.read_csv('data/raw/current_data.csv')

    X, y = clean_and_scale(df)

    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    pd.concat([X_train, y_train], axis=1).to_parquet('data/processed/train.parquet')
    pd.concat([X_val, y_val], axis=1).to_parquet('data/processed/val.parquet')
    pd.concat([X_test, y_test], axis=1).to_parquet('data/processed/test.parquet')

    print("Data successfully preprocessed and saved!")

if __name__ == "__main__":
    main()