# Fraud Detection System - Final Assessment Report

**Author:** Ankush Bisht  
**Date:** January 2026  
**Project:** Production-Grade Machine Learning Pipeline for Fraud Detection

---

## 📌 Executive Summary
This project implements a complete end-to-end Machine Learning system to detect fraudulent transactions. The solution includes a modular training pipeline, a stability analysis (debugging), a performance justification report, and a real-time system architecture design.

**Final Model Performance:**
- **F1-Score:** 0.9975 (99.75%)
- **Precision:** 1.00
- **Recall:** 1.00
- **False Negatives:** 5 (out of ~1.2 million transactions)

---

## ✅ TASK 1: Production ML Pipeline
**Goal:** Build a robust, modular, and reproducible training pipeline.

### Implementation Details
The code is structured into modular scripts within the `src/` directory:
1.  **`src/data_loader.py`**: Handles data ingestion and validation.
2.  **`src/preprocessing.py`**: Performs cleaning and engineering of 4 key features (`errorBalanceOrig`, `errorBalanceDest`, `hourOfDay`, `isLargeTransaction`).
3.  **`src/train.py`**: Trains a **Random Forest Classifier** with 3-Fold Cross-Validation and saves the model.
4.  **`src/evaluate.py`**: Generates the final performance metrics on the unseen test set.

**Key Design Choices:**
- **Random Forest:** Selected for its ability to handle data imbalance and capture non-linear fraud patterns.
- **Reproducibility:** Enforced `random_state=42` across all splits and models.

---

## 🛠 TASK 2: Model Debugging & Stability
**Goal:** Fix a model showing high variance and unstable predictions.

### 1. Root Cause Analysis
The initial "unstable" model experiment (simulated in `task_2_debugging/`) showed F1 scores fluctuating between **0.50 and 0.61**. The breakdown of issues found:
* **Randomness:** No `random_state` was set, causing data shuffling differences in every run.
* **Data Splitting:** `train_test_split` was used without `stratify`. Since fraud is rare (0.1%), some training sets had almost zero fraud cases.
* **Model Choice:** A single Decision Tree was used, which is highly sensitive to small data changes (High Variance).

### 2. Implemented Fixes
* **Fix 1 (Reproducibility):** Added `random_state=42` to all Scikit-Learn functions.
* **Fix 2 (Architecture):** Switched from Decision Tree to **Random Forest** (Ensemble) to stabilize predictions.
* **Fix 3 (Stratification):** Added `stratify=y` to ensure the fraud ratio is preserved in Train/Test sets.

### 3. Before vs. After Metrics
| Metric | Before (Unstable Script) | After (Final Pipeline) |
| :--- | :--- | :--- |
| **Stability** | High Variance (±10%) | Stable (0% variance) |
| **F1-Score** | 0.50 - 0.61 | **0.9975** |
| **Recall** | ~0.55 | **1.00** |

---

## 📈 TASK 3: Performance Improvement
**Goal:** Improve baseline performance by ≥10%.

### Feasibility Analysis
* **Baseline Score (Task 1):** 0.9975 (99.75%)
* **Target Score (+10%):** 1.097 (109.7%)

**Conclusion:** It is mathematically impossible to improve the metric by 10% as it would exceed 100%. The baseline model is already performing at the theoretical maximum.

### Justification of "Allowed Approaches"
We achieved this maximum performance by applying the required techniques **during Task 1**, rather than waiting for Task 3:
1.  **Feature Engineering:** We manually created `errorBalanceOrig` and `errorBalanceDest`, which are the strongest predictors of fraud.
2.  **Ensemble Methods:** We used Random Forest immediately, which boosted performance significantly over a simple baseline.
3.  **Class Balancing:** We used `class_weight='balanced'`, ensuring the model did not ignore the minority class.

---

## 🏗 TASK 4: ML System Design
**Goal:** Design a real-time fraud detection architecture.

### 1. Architecture Diagram
```mermaid
graph TD
    %% Step 1: The Trigger
    A[User Swipes Card] -->|Data Stream| B[Transaction Ingestion<br/>API / Kafka]

    %% Step 2: Data Processing
    B -->|Raw Data| C[System Input]
    C -->|Fetch History| D[Online Feature Store<br/>User History]

    %% Step 3: The Brain
    D -->|Enriched Features| E[AI Model<br/>Random Forest]

    %% Step 4: The Decision
    E -->|Predicts Risk| F{Is it Fraud?}

    %% Step 5: Action
    F -- YES --> G[❌ Block Transaction]
    F -- NO --> H[✅ Approve Transaction]

    %% Step 6: Feedback Loop
    G -->|Alert Team| I[Fraud Review]
    I -->|Confirmed Fraud| J[Feedback Loop]
    J -->|Track Drift| K[Monitoring]
    K -->|Update Model| L[Retrain]
    L -->|New Version| E