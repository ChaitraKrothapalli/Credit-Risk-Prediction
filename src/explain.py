import pandas as pd
import shap
import matplotlib.pyplot as plt

def print_top_features(xgb_model, X_train: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    importances = xgb_model.feature_importances_
    df_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=False).head(top_n)
    print("\nTop 10 Important Features:")
    print(df_imp)
    return df_imp

def shap_summary(xgb_model, X_test: pd.DataFrame, sample_n: int = 5000):
    X_sample = X_test.sample(n=min(sample_n, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.show()

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()
