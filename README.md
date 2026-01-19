# Credit Card Fraud Detection System

Practical machine learning pipeline for detecting fraudulent credit card transactions. The project focuses on handling highly imbalanced data, comparing multiple algorithms, and providing clear evaluation metrics for real-world decision making.

## Overview

This repository implements a full fraud-detection workflow:

- Loads and validates the dataset.
- Applies scaling and imbalance handling.
- Trains and evaluates multiple models.
- Compares results using ROC-AUC, precision, recall, and F1-score.
- Supports interactive runs and single-transaction predictions.

## Features

- Multiple models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, SVM, Neural Networks, Decision Trees, Naive Bayes, KNN.
- Ensemble voting model for stronger overall performance.
- Stratified cross-validation for reliable comparisons.
- Imbalance handling with SMOTE and related techniques.
- Visualizations for ROC curves, confusion matrices, and performance summaries.

## Dataset

The project uses the Credit Card Fraud Detection dataset (anonymized PCA features):

- 284,807 transactions, 492 fraudulent (0.17%).
- 30 features: V1-V28 plus Time and Amount.
- No missing values.

Place `creditcard.csv` in the project root or under `data/` based on your local setup.

## Installation

Requirements: Python 3.8+ and pip.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main_fraud_detection.py
```

Programmatic use:

```python
from main_fraud_detection import CreditCardFraudDetector

detector = CreditCardFraudDetector("creditcard.csv")
results = detector.run_complete_analysis()
prediction = detector.predict_new_transaction(transaction_data)
```

## Project Structure

```
credit_card_fraud_detection/
├── main_fraud_detection.py
├── fraud_dataset.py
├── data/
│   └── creditcard.csv
└── README.md
```

## Metrics Tracked

- ROC-AUC (primary for imbalance)
- Precision and recall
- F1-score
- Confusion matrix

## Notes

- Results will vary by hardware and dataset location.
- The dataset is not included in the repository.

## License

MIT License. See `LICENSE` if present in the repository.
