# Fraud Detection with XGBoost + SMOTE

This project builds a fraud detection pipeline for financial transactions using feature engineering, class-imbalance handling, and gradient boosting.

The core workflow is in [fraud-detection.ipynb](fraud-detection.ipynb), where data is prepared, modeled, and evaluated.

## Project Overview

Fraud cases are rare and costly. A model that misses fraud (false negatives) can create direct financial loss, so this project emphasizes fraud recall while keeping overall discrimination strong.

Approach used:
- Focus on transaction types where fraud occurs (`TRANSFER`, `CASH_OUT`)
- Engineer balance-consistency and time-based features
- Handle severe class imbalance with SMOTE on training data only
- Train an XGBoost binary classifier
- Evaluate using classification metrics, ROC-AUC, confusion matrix, and feature importance

## Repository Structure

- [fraud-detection.ipynb](fraud-detection.ipynb): End-to-end notebook (data prep, feature engineering, training, evaluation, insights)
- [README.md](README.md): Project documentation

## Dataset

The notebook expects a CSV at:

`/kaggle/input/internship-task-accredian/Fraud.csv`

This path is from a Kaggle environment. If running locally, update the `pd.read_csv(...)` path in [fraud-detection.ipynb](fraud-detection.ipynb#L1) to your local dataset location.

Target column:
- `isFraud` (binary)

## Methodology

### 1. Filtering and Cleaning

- Keep only `TRANSFER` and `CASH_OUT` transactions (fraud-relevant categories)
- Encode `type` as binary (`TRANSFER=0`, `CASH_OUT=1`)
- Drop leakage and high-cardinality identifiers:
	- `isFlaggedFraud`
	- `nameOrig`, `nameDest`
- Separate features and target (`X`, `Y`)

### 2. Feature Engineering

Additional predictive features created:
- `errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`
- `errorBalanceDest = oldbalanceDest + amount - newbalanceDest`
- `hour = step % 24`

These features capture transaction inconsistency and time-of-day patterns common in anomalous behavior.

### 3. Train/Test Strategy

- 80/20 split via `train_test_split`
- Apply SMOTE **only on training data** to avoid test leakage

### 4. Model

- `XGBClassifier` with:
	- `objective='binary:logistic'`
	- `eval_metric='auc'`
	- `n_estimators=100`
	- `random_state=42`

## Evaluation

The notebook reports:
- Classification report (precision, recall, f1-score)
- ROC-AUC score
- Confusion matrix
- Feature importance plot

Given fraud-risk context, recall and false negatives are key business metrics.

## Requirements

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

## How to Run

1. Open [fraud-detection.ipynb](fraud-detection.ipynb).
2. Ensure dataset path points to your `Fraud.csv`.
3. Run cells top-to-bottom.
4. Review metrics and visual outputs.

## Business Notes

- Fraud detection is an imbalanced classification problem; accuracy alone is not sufficient.
- The engineered balance-error features are particularly useful for flagging suspicious transactions.
- A threshold-tuning step can be added to optimize the precision-recall tradeoff for deployment needs.

## Future Improvements

- Add cross-validation and hyperparameter tuning
- Add PR-AUC and precision-recall curve analysis
- Calibrate probability outputs
- Export model and preprocessing pipeline for production inference
- Add automated experiment tracking
