import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'fraud_model.joblib')

def train_model():
    print("Loading processed training data...")
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    
    # Separate Features (X) and Target (y)
    X = train_df.drop(columns=['isFraud'])
    y = train_df['isFraud']

    print(f"Training data shape: {X.shape}")

    # MODEL SELECTION [Source: 9]
    # Justification: Random Forest is chosen for its robustness to outliers 
    # and ability to capture non-linear relationships in fraud data.
    # We limit depth and estimators to keep training time reasonable on a laptop.
    print("Initializing Random Forest Model...")
    clf = RandomForestClassifier(
        n_estimators=50,       # Number of trees (kept low for speed)
        max_depth=15,          # Max depth to prevent overfitting
        n_jobs=-1,             # Use all CPU cores
        random_state=42,       # Reproducibility [Source: 13]
        class_weight='balanced' # Handle imbalance (fraud is rare)
    )

    # CROSS VALIDATION [Source: 10]
    print("Running 3-Fold Cross-Validation (this might take a minute)...")
    # We use 'f1' score because accuracy is misleading in fraud detection
    cv_scores = cross_val_score(clf, X, y, cv=3, scoring='f1')
    print(f"  -> Cross-Validation F1 Scores: {cv_scores}")
    print(f"  -> Average F1 Score: {cv_scores.mean():.4f}")

    # FINAL TRAINING
    print("Training final model on full dataset...")
    clf.fit(X, y)

    # MODEL PERSISTENCE [Source: 12]
    print("Saving model...")
    joblib.dump(clf, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()