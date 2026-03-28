
import pandas as pd
import numpy as np
import joblib
import json

# === Load model and features ========
model = joblib.load('churn_model.pkl')

with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

#=== Preprocessing function ====
def preprocess(data):
    """
    Takes raw customer data and prepares it for prediction.
    Input: dict or DataFrame
    Output: encoded DataFrame ready for model
    """
    df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

    # Feature engineering
    df['EngagementScore'] = (df['HourSpendOnApp'] * df['OrderCount']) / (df['DaySinceLastOrder'] + 1)
    df['CouponDependency'] = df['CouponUsed'] / (df['OrderCount'] + 1)
    df['AddressMobility'] = df['NumberOfAddress'] / (df['Tenure'] + 1)
    df['CashbackPerOrder'] = df['CashbackAmount'] / (df['OrderCount'] + 1)
    df['IsNewCustomer'] = (df['Tenure'] <= 3).astype(int)

    # Encode categorical columns
    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode',
                'PreferedOrderCat', 'Gender', 'MaritalStatus']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Align columns with training features
    df = df.reindex(columns=feature_names, fill_value=0)

    return df

# === Scoring function ===
def score(data):
    """
    Main scoring function.
    Input: raw customer data (dict or DataFrame)
    Output: churn probability and prediction
    """
    processed = preprocess(data)
    probability = model.predict_proba(processed)[:, 1][0]
    prediction = model.predict(processed)[0]

    return {
        'churn_prediction': int(prediction),
        'churn_probability': round(float(probability), 4),
        'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.4 else 'Low'
    }

# === Test the scoring script ===
if __name__ == "__main__":
    # Example customer
    sample_customer = {
        'Tenure': 2,
        'CityTier': 1,
        'WarehouseToHome': 10,
        'HourSpendOnApp': 3,
        'NumberOfDeviceRegistered': 3,
        'SatisfactionScore': 2,
        'NumberOfAddress': 5,
        'Complain': 1,
        'OrderAmountHikeFromlastYear': 15,
        'CouponUsed': 2,
        'OrderCount': 2,
        'DaySinceLastOrder': 10,
        'CashbackAmount': 150,
        'PreferredLoginDevice': 'Mobile Phone',
        'PreferredPaymentMode': 'Debit Card',
        'PreferedOrderCat': 'Mobile',
        'Gender': 'Male',
        'MaritalStatus': 'Single'
    }

    result = score(sample_customer)
    print("=== Scoring Result ===")
    print(f"Prediction:   {'Churned' if result['churn_prediction'] == 1 else 'Retained'}")
    print(f"Probability:  {result['churn_probability']*100:.1f}%")
    print(f"Risk Level:   {result['risk_level']}")
