import joblib
import numpy as np
import pandas as pd


# Path to the saved model and its components
MODEL_PATH = "artifacts/model_data.joblib"

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]
scaler = model_data["scaler"]
features = model_data["features"]
cols_to_scale = model_data["cols_to_scale"]


def prepare_input(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type
):
    # Avoid division by zero
    loan_to_income = loan_amount / income if income > 0 else 0

    delinquent_to_loan_months = (
        delinquency_ratio / loan_tenure_months
        if loan_tenure_months > 0 else 0
    )

    input_data = {
        "age": age,
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": num_open_accounts,
        "credit_utilization_ratio": credit_utilization_ratio,

        "loan_to_income": loan_to_income,
        "delinquency_ratio": delinquency_ratio,
        "delinquent_to_loan_months": delinquent_to_loan_months,

        # Correct training feature name
        "avg_dpd_per_delinquent_month": avg_dpd_per_delinquency,

        # One-hot encoded residence type
        "residence_type_Owned": 1 if residence_type == "Owned" else 0,
        "residence_type_Rented": 1 if residence_type == "Rented" else 0,

        # One-hot encoded loan purpose
        "loan_purpose_Education": 1 if loan_purpose == "Education" else 0,
        "loan_purpose_Home": 1 if loan_purpose == "Home" else 0,
        "loan_purpose_Personal": 1 if loan_purpose == "Personal" else 0,

        # One-hot encoded loan type
        "loan_type_Unsecured": 1 if loan_type == "Unsecured" else 0,

        # Dummy values required by scaler/model
        "number_of_dependants": 1,
        "years_at_current_address": 1,
        "zipcode": 1,
        "sanction_amount": 1,
        "processing_fee": 1,
        "gst": 1,
        "net_disbursement": 1,
        "principal_outstanding": 1,
        "bank_balance_at_application": 1,
        "number_of_closed_accounts": 1,
        "enquiry_count": 1,
    }

    df = pd.DataFrame([input_data])

    # Add any missing columns expected by scaler/features
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0

    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Scale only the columns used during training
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Keep only columns expected by the model, in the exact training order
    df = df[features]

    return df


def predict(
    age,
    income,
    loan_amount,
    loan_tenure_months,
    avg_dpd_per_delinquency,
    delinquency_ratio,
    credit_utilization_ratio,
    num_open_accounts,
    residence_type,
    loan_purpose,
    loan_type
):
    input_df = prepare_input(
        age,
        income,
        loan_amount,
        loan_tenure_months,
        avg_dpd_per_delinquency,
        delinquency_ratio,
        credit_utilization_ratio,
        num_open_accounts,
        residence_type,
        loan_purpose,
        loan_type
    )

    probability, credit_score, rating = calculate_credit_score(input_df)

    return probability, credit_score, rating


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    # Logistic function
    default_probability = 1 / (1 + np.exp(-x))

    non_default_probability = 1 - default_probability

    # Convert probability to credit score
    credit_score = base_score + non_default_probability.flatten() * scale_length

    def get_rating(score):
        if 300 <= score <= 500:
            return "Poor"
        elif 500 < score <= 650:
            return "Average"
        elif 650 < score <= 750:
            return "Good"
        elif 750 < score <= 900:
            return "Excellent"
        else:
            return "Undefined"

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score[0]), rating