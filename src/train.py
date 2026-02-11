from dataclasses import dataclass
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

@dataclass
class TrainArtifacts:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_prob_xgb: pd.Series
    xgb_model: XGBClassifier

def train_models(df: pd.DataFrame, test_size: float, random_state: int, use_smote: bool) -> TrainArtifacts:
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Optional SMOTE (on train only)
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train

    # Logistic Regression (scaled)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train_res)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

    # Random Forest (unscaled)
    rf = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=random_state)
    rf.fit(X_train_res, y_train_res)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    # XGBoost (unscaled)
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
    )
    xgb.fit(X_train_res, y_train_res)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

    # Print metrics
    print("\nROC-AUC")
    print("  Logistic Regression:", round(roc_auc_score(y_test, y_prob_lr), 4))
    print("  Random Forest:      ", round(roc_auc_score(y_test, y_prob_rf), 4))
    print("  XGBoost:            ", round(roc_auc_score(y_test, y_prob_xgb), 4))

    return TrainArtifacts(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_prob_xgb=y_prob_xgb,
        xgb_model=xgb,
    )
