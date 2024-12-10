import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load("iris_svm_model.pkl")

# Streamlit app
st.title("Iris Species Predictor")
st.write("Enter the flower's characteristics below:")

# Input fields for the features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Predict button
if st.button("Predict"):
    # Prepare the input data as a 2D array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"The predicted species is: **{prediction[0]}**")
