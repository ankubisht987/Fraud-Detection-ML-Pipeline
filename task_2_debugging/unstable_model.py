import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'fraud_data.csv')

def run_unstable_experiment():
    print("Loading data...")
    # BUG 1: Loading raw data without proper cleaning pipeline
    df = pd.read_csv(DATA_PATH)
    
    # Take a small random sample to exaggerate variance (Simulating 'Unstable predictions')
    # BUG 2: No random_state here means different data every time
    df_sample = df.sample(n=50000) 
    
    X = df_sample.drop(columns=['isFraud', 'nameOrig', 'nameDest', 'type']) # Dropping type lazily
    y = df_sample['isFraud']

    # BUG 3: Splitting without 'stratify' on imbalanced data
    # This causes some runs to have 0 fraud cases in test set!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
    
    # BUG 4: Using Decision Tree (high variance model) without max_depth
    clf = DecisionTreeClassifier() 
    
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    print(f"Run Results -> Accuracy: {accuracy_score(y_test, preds):.4f} | F1 Score: {f1_score(y_test, preds):.4f}")

if __name__ == "__main__":
    print("--- RUN 1 ---")
    run_unstable_experiment()
    print("\n--- RUN 2 ---")
    run_unstable_experiment()
    print("\n--- RUN 3 ---")
    run_unstable_experiment()