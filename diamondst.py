
import pickle
import streamlit as st
import numpy as np
import pandas as pd

#load the model
trained_model = 'rf_model.pkl'
model = pickle.load(open(trained_model, 'rb'))

@st.cache_data()

def dp_prediction(carat, cut, color, clarity, depth, table, x, y, z ):
    
    pred_arr = np.array([carat, cut, color, clarity, depth, table, x, y, z])
    preds = pred_arr.reshape(1, -1)
    preds = preds.astype(float)
    
    # Make prediction
    model_prediction = model.predict(preds)
    return model_prediction
    

# Next we define the homepage of the streamlit application

def main():
    
    # front end view for the app.
    html_temp = """
    
    <div style = "background-color:green;padding:1px">
        <h1 style = "color:black;text-align:center;">
            Diamond price prediction web app 
        
    """
    
    # display the front-end aspect
    
    st.markdown(html_temp, unsafe_allow_html=True)
    
    # Next we create a box field to get user data
    
    carat = st.number_input("Carat")
    cut = st.number_input("Cut")
    color = st.number_input("Color")
    clarity = st.number_input("Clarity")
    depth = st.number_input("Depth")
    table = st.number_input("Table")
    x = st.number_input("X")
    y = st.number_input("Y")
    z = st.number_input("Z")
    
    result = ""
    
    # Create a predict button which will return the prediction when it is clicked.
    
    if st.button("Predict"):
        result = dp_prediction(carat, cut, color, clarity, depth, table, x, y, z)
        st.success("Price of diamond {}".format(result))
        
if __name__ == '__main__':
    main()
    
    
    
