# Root Cause Analysis (RCA) - Model Instability

## Issue Description
The initial model showed high variance across runs and unstable predictions. 
- **Run 1 F1-Score:** 0.6111
- **Run 2 F1-Score:** 0.5714
- **Run 3 F1-Score:** 0.5000

## Root Causes Identified
### 1. Lack of Reproducibility (Randomness)
- **Observation:** The `random_state` parameter was not set during data splitting or model initialization.
- **Impact:** Every run used a different subset of data, leading to different results.

### 2. Improper Data Splitting (Class Imbalance)
- **Observation:** The `train_test_split` function was used without `stratify=y`.
- **Impact:** Since fraud is rare (only ~0.1% of data), some random splits contained very few fraud cases in the training set, preventing the model from learning patterns effectively.

### 3. High Variance Model Selection
- **Observation:** A single `DecisionTreeClassifier` was used without depth constraints (`max_depth`).
- **Impact:** Decision Trees are prone to overfitting. Small changes in the training data resulted in a completely different tree structure.

### 4. Missing Feature Scaling/Engineering
- **Observation:** Raw features were used without engineering meaningful signals like `errorBalanceOrig`.
- **Impact:** The model struggled to find decision boundaries based on raw ID columns and unscaled amounts.

## Implemented Fixes (See `src/train.py`)
1.  **Fixed Randomness:** Set `random_state=42` in all functions.
2.  **Stratified Split:** Used `stratify=df['isFraud']` to maintain class distribution.
3.  **Model Upgrade:** Switched to `RandomForestClassifier` (an ensemble method) to reduce variance.
4.  **Feature Engineering:** Created 4 interaction features to expose fraud patterns explicitly.