# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:05:27 2024

@author: DELL
"""

import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import streamlit as st
from PIL import Image 

# Set page layout
st.set_page_config(layout="centered")
st.subheader("Welcome to the Attrition Predictor")

# Caching the model loading process for better performance
@st.cache_resource  # Use st.cache_data for Streamlit 2.x
def load_model():
    try:
        with open('model.pkl', 'rb') as model:
            return pickle.load(model)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
classifier = load_model()

# Function to make predictions
def predict_attrition(salary, jobLevel, salaryPerJobLevel, departmentLevelSalary):
    if classifier is not None:
        try:
            prediction = classifier.predict([[salary, jobLevel, salaryPerJobLevel, departmentLevelSalary]])
            return prediction
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    else:
        st.error("Model not loaded.")
        return None

# Main function to run the app
def main():
    html_temp = """
    <div style="background-color:#41B3A2;padding:10px;border-radius:10px;">
    <h2 style="color:white;text-align:center;">Streamlit Attrition Predictor App</h2>
    </div>"""
    
    job_level_list = [1, 2, 3, 4, 5]
    st.markdown(html_temp, unsafe_allow_html=True)
    
    salary = st.number_input("Salary", 30000, 150000)
    jobLevel = st.selectbox("Job Level", job_level_list)
    salaryPerJobLevel = st.number_input("Enter the average salary for the employee's job level")
    departmentLevelSalary = st.number_input("Enter average salary in the employee's department")
    
    if st.button("Predict"):
        result = predict_attrition(salary, jobLevel, salaryPerJobLevel, departmentLevelSalary)
        if result is not None:
            if result == 1:
                st.success("The employee is highly likely to leave soon.")
            else:
                st.success("There's a low probability of the employee leaving anytime soon.")
    
    elif st.button("About"):
        st.text("By Adura Kinoshi")
        st.text("Built with Streamlit.")

if __name__ == "__main__":
    main()
