import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model

def save_model(model, scaler, feature_columns, path="models/academic_model.pkl"):
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": feature_columns
    }, path)