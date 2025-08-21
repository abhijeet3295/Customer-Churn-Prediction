import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Disable GPU (optional - useful for avoiding black screen/crash issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("Customer Churn Prediction")

# Load model and encoders safely
try:
    model = tf.keras.models.load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)

    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading encoders or scaler: {e}")
    st.stop()

# User Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=650)
estimated_salary = st.number_input('Estimated Salary', value=50000.0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare Input Data
try:
    gender_encoded = label_encoder_gender.transform([gender])[0]
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_encoded],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
    })

    # Combine with one-hot encoded geography
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        st.warning(f"⚠️ Customer is **likely to churn** with a probability of **{prediction_proba:.2f}**")
    else:
        st.success(f"✅ Customer is **unlikely to churn** with a probability of **{1 - prediction_proba:.2f}**")

except Exception as e:
    st.error(f"Prediction Error: {e}")
