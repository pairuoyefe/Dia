import pandas as pd 
import streamlit as st
import numpy as np
import sklearn
import pickle



import streamlit as st
import pickle
import numpy as np


model = pickle.load(open(r'C:\Users\precc\Desktop\Deployment\Deployment.venv\Dia\Diamodel.pkl','rb'))


st.title("Diabetes Prediction App")


st.sidebar.header("Input Patient Data")

def user_input_features():
    Pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    Glucose = st.sidebar.slider("Glucose", 0, 200, 120)
    Insulin = st.sidebar.slider("Insulin", 0, 900, 80)
    BMI = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.sidebar.slider("Age", 10, 100, 33)

    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    return np.array([[Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age]])


input_data = user_input_features()


if st.button("Predict Diabetes"):
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    st.subheader(f"Prediction: {result}")
