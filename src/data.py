import pandas as pd

def load_and_prepare(csv_path: str, nrows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, nrows=nrows, low_memory=False)

    # Filter outcomes
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)

    # Feature engineering
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["installment_to_income"] = df["installment"] / (df["annual_inc"] + 1)
    df["revol_util"] = df["revol_util"] / 100.0

    selected_features = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "fico_range_low",
        "fico_range_high",
        "fico_avg",
        "loan_to_income",
        "installment_to_income",
        "revol_util",
        "installment",
        "term",
        "home_ownership",
        "verification_status",
        "purpose",
        "emp_length",
        "open_acc",
        "pub_rec",
        "delinq_2yrs",
        "total_acc",
        "inq_last_6mths",
        "target",
    ]

    df = df[selected_features].copy()

    # Fill missing numeric with median
    df = df.fillna(df.median(numeric_only=True))

    # Encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # XGBoost-safe column names
    df.columns = df.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)

    return df
