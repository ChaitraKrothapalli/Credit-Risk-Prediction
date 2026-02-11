from src.config import (
    NROWS, CSV_PATH, RANDOM_STATE, TEST_SIZE, USE_SMOTE,
    COST_FP, COST_FN, REJECT_TOP_PCT, SHAP_SAMPLE_N
)
from src.data import load_and_prepare
from src.train import train_models
from src.explain import print_top_features, shap_summary
from src.decision import loss_simulation, threshold_tuning

def main():
    print("Starting Credit Risk Project...")

    df = load_and_prepare(CSV_PATH, NROWS)
    print("Final dataset shape:", df.shape)

    artifacts = train_models(df, TEST_SIZE, RANDOM_STATE, USE_SMOTE)

    # Explainability
    print_top_features(artifacts.xgb_model, artifacts.X_train, top_n=10)
    shap_summary(artifacts.xgb_model, artifacts.X_test, sample_n=SHAP_SAMPLE_N)

    # Business + thresholds
    loss_simulation(df, artifacts.X_test, artifacts.y_test, artifacts.y_prob_xgb, reject_top_pct=REJECT_TOP_PCT)
    threshold_tuning(artifacts.y_test, artifacts.y_prob_xgb, cost_fp=COST_FP, cost_fn=COST_FN)

if __name__ == "__main__":
    main()
