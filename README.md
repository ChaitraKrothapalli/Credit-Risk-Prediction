# Credit Risk Prediction with Explainable Machine Learning

## Project Overview

This project develops an end-to-end credit risk prediction system using LendingClub loan data. The objective is to predict whether a loan will be Fully Paid or Charged Off and evaluate how the model can support financial decision-making.

The project includes:

- Financial feature engineering
- Class imbalance handling (SMOTE)
- Multiple model comparison
- Explainable AI (SHAP)
- Business loss simulation
- Threshold optimization using cost-sensitive analysis

---

## Dataset

Source: LendingClub Loan Data (2007–2018)  
Records used: 150,000 loans  

Target variable:
- 0 → Fully Paid  
- 1 → Charged Off  

---

## Methodology

### Data Preprocessing
- Filtered relevant loan statuses
- Handled missing values using median imputation
- One-hot encoded categorical variables
- Sanitized feature names for XGBoost compatibility

### Feature Engineering
Engineered financially meaningful predictors:

- FICO average score  
- Loan-to-Income ratio  
- Installment-to-Income ratio  
- Revolving utilization normalization  

These features better capture repayment stress and creditworthiness.

### Models Trained
- Logistic Regression  
- Random Forest  
- XGBoost  

SMOTE was applied to address class imbalance during training.

---

## Model Performance

| Model                | ROC-AUC |
|----------------------|---------|
| Logistic Regression  | ~0.70   |
| Random Forest        | ~0.73   |
| XGBoost              | ~0.73   |

XGBoost provided stable performance and was selected for further analysis.

---

## Explainability

Feature importance and SHAP analysis identified key drivers of default risk:

- Loan term (60 months)  
- Home ownership status  
- Credit inquiries in last 6 months  
- Verification status  
- Interest rate  
- Public records  

These results align with financial risk theory.

---

## Business Impact Simulation

A portfolio strategy simulation was conducted:

Rejecting the top 10% highest predicted risk loans could potentially prevent approximately $19 million in losses.

This demonstrates how predictive modeling can inform risk management decisions.

---

## Threshold Optimization

Rather than using a default probability threshold of 0.5:

- Best F1-score threshold: 0.25  
- Cost-sensitive optimal threshold: 0.10  

Cost-based tuning significantly improved recall for default detection (≈92%), which is critical in financial risk management.

---

## Project Structure

Credit-Risk-Prediction/
│
├── src/
│ ├── config.py
│ ├── data.py
│ ├── train.py
│ ├── explain.py
│ ├── decision.py
│ └── main.py
├── README.md


---

## How to Run

1. Create a virtual environment:


2. Install dependencies:


3. Run the pipeline:

## Future Improvements

- Time-based train/test split to avoid leakage  
- Hyperparameter tuning  
- Probability calibration  
- Cross-validation  
- Deployment as API using FastAPI  

---

