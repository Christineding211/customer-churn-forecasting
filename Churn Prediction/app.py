
import gradio as gr
import pandas as pd
import joblib
import os

# 1. Check model and scaler files
assert os.path.exists("logistic_model.joblib"), "❌ Model file not found."
assert os.path.exists("scaler.joblib"), "❌ Scaler file not found."

# 2. Load model and scaler
model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")

# 3. Prediction function
def predict_churn(tenure, monthly_charges, total_charges,
                  gender, senior, partner, dependents,
                  contract, internet, payment,
                  num_service, charge_bin, tenure_group):
    try:
        # Standardize numeric features
        df_input = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Log_TotalCharges": total_charges
        }])
        df_input["Log_TotalCharges"] = pd.to_numeric(df_input["Log_TotalCharges"], errors='coerce').fillna(0)
        df_scaled = scaler.transform(df_input[["tenure", "MonthlyCharges", "Log_TotalCharges"]])
        df = pd.DataFrame(df_scaled, columns=["tenure_scaled", "MonthlyCharges_scaled", "Log_TotalCharges_scaled"])

        # Encode categorical and engineered features
        df["gender"] = 1 if gender == "Male" else 0
        df["SeniorCitizen"] = int(senior)
        df["Partner"] = int(partner)
        df["Dependents"] = int(dependents)
        df["high_charges"] = 1 if monthly_charges > 70 else 0
        df["Contract_One year"] = 1 if contract == "One year" else 0
        df["Contract_Two year"] = 1 if contract == "Two year" else 0
        df["InternetService_DSL"] = 1 if internet == "DSL" else 0
        df["InternetService_Fiber optic"] = 1 if internet == "Fiber optic" else 0
        df["PaymentMethod_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
        df["PaymentMethod_Electronic check"] = 1 if payment == "Electronic check" else 0
        df["PaymentMethod_Mailed check"] = 1 if payment == "Mailed check" else 0
        df["num_service"] = int(num_service)
        df["charge_bin_ord"] = int(charge_bin)
        df["tenure_group_ord"] = int(tenure_group)

        # Ensure correct feature order
        if hasattr(model, 'feature_names_in_'):
            df = df[model.feature_names_in_]

        prob = model.predict_proba(df)[0][1]
        return f"📉 Predicted Churn Probability: {prob:.2%}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# 4. Gradio UI
ui = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Monthly Charges ($)"),
        gr.Number(label="Total Charges ($)"),
        gr.Radio(["Male", "Female"], label="Gender"),
        gr.Radio([0, 1], label="Senior Citizen (0 = No, 1 = Yes)"),
        gr.Radio([0, 1], label="Has Partner (0 = No, 1 = Yes)"),
        gr.Radio([0, 1], label="Has Dependents (0 = No, 1 = Yes)"),
        gr.Radio(["One year", "Two year", "Month-to-month"], label="Contract Type"),
        gr.Radio(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Radio(["Credit card (automatic)", "Electronic check", "Mailed check"], label="Payment Method"),
        gr.Slider(0, 6, step=1, label="Number of Services Used"),
        gr.Slider(0, 4, step=1, label="Charge Bin (Ordinal)"),
        gr.Slider(0, 3, step=1, label="Tenure Group (Ordinal)")
    ],
    outputs="text",
    title="📊 Customer Churn Prediction",
    description="Enter customer information to estimate their probability of churn."
)

# 5. Launch the app
if __name__ == "__main__":
    ui.launch()
