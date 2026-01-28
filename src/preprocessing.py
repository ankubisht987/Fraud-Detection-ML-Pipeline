import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train_processed.csv')
PROCESSED_TEST_PATH = os.path.join(BASE_DIR, 'data', 'test_processed.csv')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder.joblib')

def feature_engineering(df):
    """
    Creates 4 meaningful features as required by Task 1.
    """
    print("  -> Engineering features...")
    
    # Feature 1: Error in Origin Account (Difference between actual change and transaction amount)
    # Rationale: Fraudsters often manipulate the backend so balances don't update correctly.
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']

    # Feature 2: Error in Destination Account
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

    # Feature 3: Hour of the Day (Step is in hours)
    # Rationale: Fraud often happens at odd hours.
    df['hourOfDay'] = df['step'] % 24

    # Feature 4: Large Transaction Flag
    # Rationale: Large amounts are riskier.
    df['isLargeTransaction'] = (df['amount'] > 200000).astype(int)

    return df

def preprocess_data():
    print("Loading split data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    # 1. CLEANING & VALIDATION [Source: 7]
    print("Cleaning data...")
    
    # Drop irrelevant columns (IDs don't help general logic)
    cols_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    train = train.drop(columns=cols_to_drop, errors='ignore')
    test = test.drop(columns=cols_to_drop, errors='ignore')

    # 2. FEATURE ENGINEERING [Source: 8]
    train = feature_engineering(train)
    test = feature_engineering(test)

    # 3. ENCODING CATEGORICAL DATA ('type')
    # We must treat 'type' (TRANSFER, CASH_OUT, etc.) numerically.
    print("Encoding categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit on TRAIN, transform both (Prevent Data Leakage!)
    type_encoded_train = encoder.fit_transform(train[['type']])
    type_encoded_test = encoder.transform(test[['type']])

    # Convert to DataFrames
    type_cols = encoder.get_feature_names_out(['type'])
    train_encoded_df = pd.DataFrame(type_encoded_train, columns=type_cols, index=train.index)
    test_encoded_df = pd.DataFrame(type_encoded_test, columns=type_cols, index=test.index)

    # Join back and drop original 'type'
    train = pd.concat([train.drop(columns=['type']), train_encoded_df], axis=1)
    test = pd.concat([test.drop(columns=['type']), test_encoded_df], axis=1)

    # Save the encoder for later (Model Persistence requirement)
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
    joblib.dump(encoder, ENCODER_PATH)

    # Save processed data
    print("Saving processed data...")
    train.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test.to_csv(PROCESSED_TEST_PATH, index=False)
    
    print(f"✅ Preprocessing Complete!")
    print(f"Processed Train Shape: {train.shape}")
    print(f"Processed Test Shape: {test.shape}")
    print(f"New Features: errorBalanceOrig, errorBalanceDest, hourOfDay, isLargeTransaction")

if __name__ == "__main__":
    preprocess_data()