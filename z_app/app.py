import streamlit as st
import pandas as pd
import joblib

# Load model objects

model = joblib.load("models/burnout_final_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
academic_model = joblib.load("models/academic_model.pkl")
academic_scaler = joblib.load("models/academic_scaler.pkl")
academic_features_columns = joblib.load("models/academic_feature_columns.pkl")

st.title("Student Burnout & Academic Risk Prediction")

# User Inputs
age = st.number_input("Age")
study_hours = st.number_input("Study Hours per Day")
sleep_hours = st.number_input("Sleep Hours per Day")
attendance = st.number_input("Attendance Percentage")
assignment_delay = st.number_input("Assignment Delay Days")
screen_time = st.number_input("Screen Time Hours")
physical_activity = st.number_input("Physical Activity Hours")
cgpa = st.number_input("CGPA")
gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):

    input_data = {
        "age": age,
        "study_hours_per_day": study_hours,
        "sleep_hours_per_day": sleep_hours,
        "attendance_percentage": attendance,
        "assignment_delay_days": assignment_delay,
        "screen_time_hours": screen_time,
        "physical_activity_hours": physical_activity,
        "cgpa": cgpa,
        "gender": gender
    }

    input_df = pd.DataFrame([input_data])

    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    st.success(f"Predicted Burnout Level: {prediction[0]}")

    # Academic Risk Prediction

    academic_input_df = pd.DataFrame([input_data])
    academic_input_df = pd.get_dummies(academic_input_df, drop_first=True)
    academic_input_df = academic_input_df.reindex(columns=academic_features_columns, fill_value=0)

    academic_scaled = academic_scaler.transform(academic_input_df)
    academic_prediction = academic_model.predict(academic_scaled)

    st.info(f"Predicted Academic Risk: {academic_prediction[0]}")