#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 12:03:10 2023

@author: GraceTee
"""

import numpy as np
import pickle
import streamlit as st



loaded_model = pickle.load(open('trained_cancer_model.sav', 'rb'))

#Creating a prediction function 

def cancer_prediction(input_data):
    
    #Changing the input data to numpy array
    data_numpy_array = np.asarray(input_data)
    
    #reshape the array 
    data_reshaped = data_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(data_reshaped)
    print(prediction)
    
    if (prediction[0] == 'B'):
        return "Prediction: The tumor is benign."
    else:
        return "Prediction: The tumor is malignant."
        print("The tumor is malignant")
        
        
def main():
    # Title
    st.title('Breast Tissue Cancer Prediction')

    # Display the input form
    st.header("Input Data")

    # Features related to mean values
    st.subheader("Mean Values")
    radius_mean = st.number_input("Radius Mean")
    texture_mean = st.number_input("Texture Mean")
    perimeter_mean = st.number_input("Perimeter Mean")
    area_mean = st.number_input("Area Mean")
    smoothness_mean = st.number_input("Smoothness Mean")
    compactness_mean = st.number_input("Compactness Mean")
    concavity_mean = st.number_input("Concavity Mean")
    concave_points_mean = st.number_input("Concave Points Mean")
    symmetry_mean = st.number_input("Symmetry Mean")
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean")

    # Features related to standard errors
    st.subheader("Standard Errors")
    radius_se = st.number_input("Radius SE")
    texture_se = st.number_input("Texture SE")
    perimeter_se = st.number_input("Perimeter SE")
    area_se = st.number_input("Area SE")
    smoothness_se = st.number_input("Smoothness SE")
    compactness_se = st.number_input("Compactness SE")
    concavity_se = st.number_input("Concavity SE")
    concave_points_se = st.number_input("Concave Points SE")
    symmetry_se = st.number_input("Symmetry SE")
    fractal_dimension_se = st.number_input("Fractal Dimension SE")

    # Features related to worst values
    st.subheader("Worst Values")
    radius_worst = st.number_input("Radius Worst")
    texture_worst = st.number_input("Texture Worst")
    perimeter_worst = st.number_input("Perimeter Worst")
    area_worst = st.number_input("Area Worst")
    smoothness_worst = st.number_input("Smoothness Worst")
    compactness_worst = st.number_input("Compactness Worst")
    concavity_worst = st.number_input("Concavity Worst")
    concave_points_worst = st.number_input("Concave Points Worst")
    symmetry_worst = st.number_input("Symmetry Worst")
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst")

    # Create a list of input data
    input_data = [
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]

    # Predict button
    if st.button("Breast Tissue Test Result"):
        prediction_result = cancer_prediction(input_data)
        st.success(prediction_result)

if __name__ == "__main__":
    main()
    
    
