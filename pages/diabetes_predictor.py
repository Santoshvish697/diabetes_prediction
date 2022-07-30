import streamlit as st
import pandas as pd
import os
from Final_ML_Analysis_LR_DEC_RF_SVM import *
import fontstyle
from datetime import date


# pickle_in = open('rf_classifier.pkl','rb')
# classifier = pickle.load(pickle_in)

pickle_svc = open('svc_classifier.pkl','rb')
svc = pickle.load(pickle_svc)

st.title('Diabetes Predictor')
name = st.text_input("Name:")


ques = st.radio("Gender", ('Male','Female'))

pregnancy = st.number_input("No. of times pregnant:",min_value=0.0)
glucose = st.number_input("Plasma Glucose Concentration :",value=0.0,min_value=0.0)
bp =  st.number_input("Diastolic blood pressure (mm Hg):",value = 0.0, min_value = 0.0)
skin = st.number_input("Triceps skin fold thickness (mm):",value = 0.0, min_value=0.0)
insulin = st.number_input("2-Hour serum insulin (mu U/ml):",value = 0.0,min_value=0.0)
bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):",value = 0.0,min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function:",value = 0.0,min_value=0.0)
age = st.number_input("Age:",value = 21,min_value = 21)



sc_x = StandardScaler()
input_arr = np.array([pregnancy,glucose,bp,skin,insulin,bmi,dpf,age]).reshape(-1,1)
sc_arr = input_arr.reshape(1,8)

submit = st.button('Predict')
if submit:
        prediction = svc.predict(sc_arr)
        if prediction == 0:
            st.write ('Congratulation',name,'You are not diabetic!')
        else:
            st.write("***We are really sorry to say but it seems like you are Diabetic.***")

