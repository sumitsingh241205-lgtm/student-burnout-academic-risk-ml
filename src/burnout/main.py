import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.burnout.data_preprocessing import load_data, preprocess_data
from src.burnout.train_model import train_model, save_model
from src.burnout.evaluate import evaluate_model

def main():

    df = load_data("data/raw/student_burnout_dataset_350.csv")

    X_train, X_test, y_train, y_test, feature_columns = preprocess_data(df)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model, feature_columns)

 
if __name__ == "__main__":
    main()