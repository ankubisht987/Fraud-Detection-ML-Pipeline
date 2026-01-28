import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.joblib')

def evaluate_model():
    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)
    
    # Separate Features (X) and Target (y)
    X_test = test_df.drop(columns=['isFraud'])
    y_test = test_df['isFraud']
    
    print("Loading saved model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found! Run train.py first.")
        
    clf = joblib.load(MODEL_PATH)
    
    print("Making predictions...")
    y_pred = clf.predict(X_test)
    
    print("\n" + "="*40)
    print("FINAL MODEL EVALUATION REPORT")
    print("="*40)
    
    # METRICS [Source: 11]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*40)
    print("✅ Evaluation Complete. Task 1 Pipeline is FINISHED.")

if __name__ == "__main__":
    evaluate_model()