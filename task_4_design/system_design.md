# Task 4: ML System Design - Real-Time Fraud Detection

## 1. System Architecture Diagram

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