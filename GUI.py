import streamlit as st
import numpy as np
import pickle as p
# loading my model
lr=p.load(open('trained_model.sav','rb'))
# create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the comma seperated feature set...")
input_df = st.text_input('Input All features')
input_df_lst = input_df.split(',')
submit = st.button("Submit")

if submit:
    # get input feature values
    features = np.array(input_df_lst, dtype=np.float64)
    # make prediction
    prediction = lr.predict(features.reshape(1,-1))
    # display result
    if prediction[0] == 0:
        st.write("Legitimate transaction")
    else:
        st.write("Fraudulent transaction")