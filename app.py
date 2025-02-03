#!/usr/bin/env python
# coding: utf-8

import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load trained model
filename = 'modell.pkl'
with open(filename, 'rb') as file:
    loaded_model = pickle.load(file)

st.title('Tips Prediction App')
st.subheader('Please enter your data:')

# Load dataset for reference
df = pd.read_csv('tips.csv')

# Feature names used in model training
model_features = loaded_model.feature_names_in_  # Ensure these match

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['sex', 'smoker', 'day', 'time']

    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])  # Ensure encoding is consistent

    # Ensure only trained features are used
    df_preprocessed = df.reindex(columns=model_features, fill_value=0)

    # Make prediction
    prediction = loaded_model.predict(df_preprocessed)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')

    # Display result
    st.subheader('Lifestyle Change Prediction:')
    st.write(prediction_text)
