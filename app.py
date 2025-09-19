import streamlit as st
import pickle
import pandas as pd

# Load trained model and feature list
model = pickle.load(open("model.pkl", "rb"))
model_features = pickle.load(open("model_features.pkl", "rb"))

# Page setup
st.set_page_config(page_title="Employee Promotion Predictor", layout="centered")
st.title("🎯 Employee Promotion Prediction")
st.write("Fill in the employee details below to predict promotion likelihood.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    employee_id = st.text_input("Employee ID", value="E123")
    age = st.number_input("Age", min_value=18, max_value=60, value=30)
    length_of_service = st.number_input("Length of Service", min_value=0, max_value=40, value=5)
    avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100, value=50)
    no_of_trainings = st.number_input("Number of Trainings", min_value=0, max_value=20, value=2)
    kpi_met = st.selectbox("KPIs Met >80%", ["Yes", "No"])
    awards_won = st.selectbox("Awards Won?", ["Yes", "No"])
with col2:
    previous_year_rating = st.selectbox("Previous Year Rating", [1, 2, 3, 4, 5])
    department = st.selectbox("Department", [
        "Analytics", "Finance", "HR", "Legal", "Operations",
        "Procurement", "R&D", "Sales & Marketing", "Technology"
    ])
    education = st.selectbox("Education Level", [
        "Below Secondary", "Bachelor's", "Master's & above"
    ])
    gender = st.selectbox("Gender", ["Male", "Female"])
    region = st.selectbox("Region", [
        "region_1", "region_2", "region_3", "region_4", "region_5",
        "region_6", "region_7", "region_8", "region_9", "region_10"
    ])

# Predict button
if st.button("🔍 Predict"):
    # Raw input dictionary
    input_dict = {
        "employee_id": employee_id,
        "age": age,
        "length_of_service": length_of_service,
        "avg_training_score": avg_training_score,
        "no_of_trainings": no_of_trainings,
        "KPIs_met >80%": 1 if kpi_met == "Yes" else 0,
        "awards_won?": 1 if awards_won == "Yes" else 0,
        "previous_year_rating": previous_year_rating,
        "department": department,
        "education": education,
        "gender": "f" if gender == "Female" else "m",
        "region": region
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode categorical features
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns and reorder to match training
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_features]

    # Predict
    prediction = model.predict(input_encoded)[0]
    try:
        prob = model.predict_proba(input_encoded)[0][1]
        st.write(f"📊 Promotion likelihood: **{prob:.2%}**")
    except:
        pass

    if prediction == 1:
        st.success("✅ Employee is likely to be promoted 🎉")
    else:
        st.error("❌ Employee is not likely to be promoted")