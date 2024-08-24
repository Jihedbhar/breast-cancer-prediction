import pickle
import numpy as np
import streamlit as st
import os
from tensorflow import keras

# Define the folder path directly
folder_path = r'C:\projects\B_cancer'
model_path = os.path.join(folder_path, 'model.pkl')
scaler_path = os.path.join(folder_path, 'scaler.pkl')

# Load the model and scaler
with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Breast Cancer Prediction System By Jihed Bhar")

# Default input data
default_values = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,
                  0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,
                  12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# Input fields for all attributes with default values
st.header("Enter the details:")

attributes = [
    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 
    'Mean Compactness', 'Mean Concavity', 'Mean Concave Points', 'Mean Symmetry', 
    'Mean Fractal Dimension', 'Radius SE', 'Texture SE', 'Perimeter SE', 
    'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 
    'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Worst Radius', 
    'Worst Texture', 'Worst Perimeter', 'Worst Area', 'Worst Smoothness', 
    'Worst Compactness', 'Worst Concavity', 'Worst Concave Points', 
    'Worst Symmetry', 'Worst Fractal Dimension'
]

# Create input fields with default values
input_data = [st.number_input(attr, value=default_values[i]) for i, attr in enumerate(attributes)]

# Convert the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array for a single prediction
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data
input_data_std = scaler.transform(input_data_reshaped)

# Make a prediction
prediction = model.predict(input_data_std)
prediction_label = np.argmax(prediction)

# Display the result
if prediction_label == 0:
    st.write("The tumor is **Malignant**.")
else:
    st.write("The tumor is **Benign**.")
