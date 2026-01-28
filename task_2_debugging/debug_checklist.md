# Machine Learning Debug Checklist

## 1. Data Integrity & Splitting
- [x] Is `random_state` fixed for reproducibility?
- [x] Is the target variable distribution preserved? (Use `stratify=y` for classification)
- [x] Are there duplicates or data leakage between Train and Test sets?

## 2. Feature Engineering
- [x] Are categorical variables encoded correctly (OneHot/Label)?
- [x] Are numerical features scaled if necessary (e.g., for SVM/KNN)?
- [x] Have ID columns (non-predictive) been removed?

## 3. Model Stability
- [x] Does the model show high variance? (Check Cross-Validation scores)
- [x] Is the model overfitting? (Train score >>> Test score)
- [x] Is the class imbalance handled? (Use `class_weight='balanced'` or SMOTE)

## 4. Evaluation
- [x] Are we using the right metric? (Accuracy is bad for fraud; use F1-Score or AUC)
- [x] Is the confusion matrix analyzed for False Positives vs False Negatives?