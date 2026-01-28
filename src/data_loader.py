import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Define Paths
# This ensures code runs on any machine regardless of OS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'fraud_data.csv')
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')

def load_and_split_data():
    """
    Loads raw data, performs basic validation, splits into train/test,
    and saves them separately.
    """
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found at {DATA_PATH}. Did you move it to the 'data' folder?")

    df = pd.read_csv(DATA_PATH)
    
    # VALIDATION: Check for empty data
    # Requirement: Data validation 
    if df.empty:
        raise ValueError("The dataset is empty!")
    
    print(f"Data loaded successfully. Shape: {df.shape}")

    # SPLIT DATA
    # Requirement: Reproducibility (random seeds) 
    # Stratify by 'isFraud' to handle imbalance (very important for fraud data)
    print("Splitting data...")
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['isFraud'])
    
    # Save the splits
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    
    print(f"✅ Data saved!")
    print(f"Train set: {train.shape}")
    print(f"Test set: {test.shape}")

if __name__ == "__main__":
    load_and_split_data()