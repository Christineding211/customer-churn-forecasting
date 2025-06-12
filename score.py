import pandas as pd
import joblib

# 載入模型與 scaler
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")

# 預測函數（供 API 使用）
def score(data: pd.DataFrame) -> float:
    df_scaled = scaler.transform(data[["tenure", "MonthlyCharges", "Log_TotalCharges"]])
    df = pd.DataFrame(df_scaled, columns=["tenure_scaled", "MonthlyCharges_scaled", "Log_TotalCharges_scaled"])
    for col in data.columns:
        if col not in ["tenure", "MonthlyCharges", "Log_TotalCharges"]:
            df[col] = data[col].values
    return model.predict_proba(df)[0][1]
