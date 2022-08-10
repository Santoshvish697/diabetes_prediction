import streamlit as st
import pandas as pd
import os
import fontstyle
import datetime
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np


#Add Background
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

add_bg_from_local('pages/images/back.png')    

#Save ML Model 
pickle_svc = open('svc_classifier.pkl','rb')
svc = pickle.load(pickle_svc)

x = [datetime.datetime.now() + datetime.timedelta(days=i) for i in range(30)]

df_glucose = pd.DataFrame({'DateTime': x,
'Glucose': 0},columns = ["DateTime","Glucose"],index = [x for x in range(30)])

def append_df(df,count,val):
    count = count + 1
    df.loc[count,'Glucose'] = val
    plot_glucose(df)
    return count

def plot_glucose(df):
    fig = px.line(df,x = "DateTime",y = "Glucose")
    fig.write_html("pages/graph.html")
    return

    
#Global Variable
cur = 0
count = -1
flag = 1

#Form Elements
st.title('Diabetes Predictor')
name = st.text_input("Name:")


ques = st.radio("Gender", ('Male','Female'))

pregnancy = st.number_input("No. of times pregnant:",min_value=0.0)
glucose = st.number_input("Plasma Glucose Concentration [100.00-200.00] Range :",value=0.0,min_value=0.0)
bp =  st.number_input("Diastolic blood pressure [20.00-140.00] mmHg Range:",value = 0.0, min_value = 0.0)
skin = st.number_input("Triceps skin fold thickness Upto 110.00 mm Range:",value = 0.0, min_value=0.0)
insulin = st.number_input("2-Hour serum insulin Upto 745.00 mu U/ml Range:",value = 0.0,min_value=0.0)
bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):",value = 0.0,min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function [0-3] Range:",value = 0.0,min_value=0.0)
age = st.number_input("Age:",value = 21,min_value = 21)



sc_x = StandardScaler()
input_arr = np.array([pregnancy,glucose,bp,skin,insulin,bmi,dpf,age]).reshape(-1,1)
sc_arr = input_arr.reshape(1,8)

submit = st.button('Predict')
if submit:
    count = count+1
    prediction = svc.predict(sc_arr)
    if prediction == 0:
        st.write ('Congratulation',name,'You are not diabetic!')
    else:
        st.write("***We are really sorry to say but it seems like you are Diabetic.***")
    
    cur = append_df(df_glucose,count,glucose) 
    count = count+1

