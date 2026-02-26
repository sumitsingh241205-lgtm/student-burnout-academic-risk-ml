import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd

def predict_new_student(input_dict):

    model = joblib.load("models/burnout_final_model.pkl")
    

    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df, drop_first=True)

    feature_columns = joblib.load("models/feature_columns.pkl")
    df = df.reindex(columns=feature_columns, fill_value=0)

    

    prediction = model.predict(df)

    return prediction
if __name__ == "__main__":

    sample_student = {
        "age": 21,
        "study_hours_per_day": 5,
        "sleep_hours_per_day": 6,
        "attendance_percentage": 80,
        "assignment_delay_days": 3,
        "screen_time_hours": 6,
        "physical_activity_hours": 1,
        "cgpa": 7.5,
        "gender_Male": 1
    }

    result = predict_new_student(sample_student)

    print("Predicted Burnout Level:", result)