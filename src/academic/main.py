import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.academic.preprocessing import load_data, preprocess_data
from src.academic.train_model import train_model, save_model
from src.academic.evaluate import evaluate_model

def main():
    df = load_data("data/raw/student_academic_dataset.csv")

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols = preprocess_data(df)

    model = train_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test)

    save_model(model, scaler, feature_cols)

if __name__ == "__main__":
    main()