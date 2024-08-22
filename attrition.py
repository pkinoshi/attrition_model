# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:05:27 2024

@author: DELL
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(layout="centered")
st.subheader("Welcome to the Attrition predictor")

model = open("model.pkl", "rb") #open and read model in binary format
classifier = pickle.load(model)



def predict_attrition(salary, jobLevel, salaryPerJobLevel, departmentLevelSalary):
    prediction = classifier.predict([[salary, jobLevel, salaryPerJobLevel, departmentLevelSalary]])
    print(prediction)
    return prediction


def main():
    html_temp = """
    <div style="background-color:#41B3A2;padding:10px";border-radius:50%>
    <h2 style="color:white;text-align:center;">Streamlit Attrition Predictor App</h2>
    </div>"""
    
    job_level_list = [1,2,3,4,5]
    st.markdown(html_temp, unsafe_allow_html=True)
    salary = st.number_input("Salary", 30000, 150000)#"Enter actual salary of employee"
    jobLevel = st.selectbox("Job Level", job_level_list)#, "Enter job level of employee"
    salaryPerJobLevel = st.number_input("Enter the average salary for the employee's job level")
    departmentLevelSalary = st.number_input("Enter average salary in the employee's department")
    
    if st.button("Predict"):
        result = predict_attrition(salary, jobLevel, salaryPerJobLevel, departmentLevelSalary)
        if result == 1:
            st.success("The employee is highly likely to leave soon.")
        else:
            st.success("There's a low probability of the employee leaving anytime soon.")
    
    elif st.button("About"):
        st.text("By Adura Kinoshi")
        st.text("Built with streamlit.")
        
if __name__ == "__main__":
    main()
