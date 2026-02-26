import joblib
import pandas as pd

def load_model(path="models/academic_model.pkl"):
    return joblib.load(path)

def predict(data_dict, model_package):
    model = model_package["model"]
    scaler = model_package["scaler"]
    features = model_package["features"]

    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=features, fill_value=0)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)

    return prediction[0]