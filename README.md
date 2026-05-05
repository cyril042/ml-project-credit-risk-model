# Lauki Finance: Credit Risk Modeling

## Overview
This project develops an end-to-end machine learning solution for credit risk assessment.

The system:
- predicts borrower default probability
- generates credit scores (300–900)
- assigns credit ratings
- supports real-time risk assessment through an interactive dashboard

## Live Demo
[Streamlit App](https://cyril-ml-project-credit-risk-model.streamlit.app/)

## Business Problem
Financial institutions need reliable methods for evaluating borrower risk before loan approval.

This project demonstrates how machine learning can support:
- credit underwriting
- loan approval decisions
- portfolio risk monitoring
- customer segmentation

## Features Engineered
- Loan-to-Income Ratio
- Delinquency Ratio
- Credit Utilization
- Average DPD
- Repayment History
- Borrower Demographics

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git

  ## Model Performance

After feature engineering, class imbalance handling, and hyperparameter optimization:

### Best Production Model
**Optuna-Tuned Logistic Regression + SMOTE-Tomek**

| Metric | Score |
|--------|-------|
| Accuracy | 93% |
| Cross-Validated Macro F1 | 0.945 |
| Weighted F1 | 0.94 |
| Default-Class Recall | 94% |
| Default-Class Precision | 57% |

### Business Impact
The model successfully identifies high-risk borrowers with high sensitivity:

- Detects **94% of potential defaults**
- Maintains **93% overall prediction accuracy**
- Produces interpretable credit scores from **300–900**
- Supports real-time underwriting decisions via deployed dashboard

### Deployment
The final model was saved using joblib and deployed using:
- GitHub
- Streamlit Cloud

## Project Structure

ml-project-credit-risk-model/
├── main.py
├── prediction_helper.py
├── artifacts/
├── requirements.txt
├── notebooks/
└── README.md
