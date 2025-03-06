import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import tensorflow as tf

# Loading the model
model = tf.keras.models.load_model('model.keras')

with open('geo_enc.pkl','rb') as file:
    geo_enc = pickle.load(file)
    
with open('gen_enc.pkl','rb') as file:
    gen_enc = pickle.load(file)
    
with open('scaler.pkl','rb') as file:
    scalar = pickle.load(file)
    
## The app
st.title('Customer CHURN Prediction')

# Collecting user input
geography = st.selectbox('Geography', geo_enc.categories_[0])
gender = st.selectbox('Gender', gen_enc.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare the input from the collected data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [gen_enc.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

# One hot encode Geography and combine it to the dataframe
geo_encoded = geo_enc.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=geo_enc.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# scale the input data
input_data_scaled = scalar.transform(input_data)

# Prediction and result
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba: .2f}')

if prediction_proba > 0.5:
    st.write("Customer is likely to CHURN.")
else:
    st.write("Customer is not likely to CHURN.")