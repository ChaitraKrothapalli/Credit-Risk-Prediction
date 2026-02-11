import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report

def loss_simulation(df_full: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, y_prob: np.ndarray, reject_top_pct: float):
    # uses avg loan amount from df_full
    avg_loan = df_full["loan_amnt"].mean()

    temp = X_test.copy()
    temp["actual_default"] = y_test.values
    temp["predicted_risk"] = y_prob

    temp = temp.sort_values("predicted_risk", ascending=False)
    cutoff = int(reject_top_pct * len(temp))
    rejected = temp.iloc[:cutoff]

    estimated_loss_prevented = rejected["actual_default"].sum() * avg_loan
    print(f"\nEstimated Loss Prevented by Rejecting Top {int(reject_top_pct*100)}% High-Risk Loans: ${estimated_loss_prevented:,.2f}")

def threshold_tuning(y_test: pd.Series, y_prob: np.ndarray, cost_fp: int = 1, cost_fn: int = 10):
    thresholds = np.linspace(0.05, 0.95, 19)

    # Best F1
    best_f1, best_t_f1 = -1, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1, best_t_f1 = f1, t

    print("\nBest F1 Threshold:", best_t_f1, "F1:", round(best_f1, 4))
    preds_f1 = (y_prob >= best_t_f1).astype(int)
    print("Confusion Matrix (F1 threshold):\n", confusion_matrix(y_test, preds_f1))

    # Best cost
    best_cost, best_t_cost = float("inf"), 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost, best_t_cost = cost, t

    print("\nBest Cost Threshold:", best_t_cost, "Cost:", best_cost)
    preds_cost = (y_prob >= best_t_cost).astype(int)
    print("Confusion Matrix (Cost threshold):\n", confusion_matrix(y_test, preds_cost))
    print("\nClassification Report (Cost threshold):")
    print(classification_report(y_test, preds_cost))
