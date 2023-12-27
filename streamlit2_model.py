# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:25:13 2023

@author: Along
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from joblib import load
import streamlit as st


# Import Model

model = load('streamlit2_model.joblib')

def preprocess_data(df):
    
    processed_df = df.copy()
    
    # Perform label encoding and one-hot encoding
    categorical_cols = ['country', 'year','location_type', 'cellphone_access','gender_of_respondent',
       'relationship_with_head', 'marital_status', 'education_level',
       'job_type']
    for col in categorical_cols:
        if col in processed_df.columns:
            lb = LabelEncoder()
            processed_df[col] = lb.fit_transform(processed_df[col])
            
    for i in processed_df.columns:
        if processed_df[i].dtypes != 'object':
            scalar = MinMaxScaler()
            processed_df[[i]] = scalar.fit_transform(processed_df[[i]])
           
    return processed_df
# Function to preprocess input data
def preprocessor(input_df):
    # Preprocess categorical and numerical columns separately
    input_df = preprocess_data(input_df)
    return input_df



# Streamlit app
def main():
    st.title('Financial Inclusion Prediction')
    st.write('Welcome to  my prediction app. This App helps predit if the persona has a bank account or not.')

    input_data = {}  # Dictionary to store user input data
    col1, col2 = st.columns(2)  # Split the interface into two columns

    with col1:
        # Collect user inputs for country and some financial indicators
        input_data['country'] = st.selectbox('Country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
        input_data['year'] = st.number_input('What year?', min_value=0)
        input_data['location_type'] = st.selectbox('Location Type', ['Rural', 'Urban'])
        input_data['cellphone_access'] = st.selectbox('Cellphone Access', ['Yes', 'No'])
        input_data['household_size'] = st.number_input('Household Size', min_value=0)
        input_data['age_of_respondent'] = st.number_input('Age of Respondent', min_value=0)
        

        

    with col2:
        # Collect user inputs for other indicators
        
        input_data['relationship_with_head'] = st.selectbox('Relationship with head', ['Spouse', 'Head of Household', 'Other relative', 'Child', 'Parent',
       'Other non-relatives'])
        input_data['marital_status'] = st.selectbox('Marital Status', ['Married/Living together', 'Widowed', 'Single/Never Married',
       'Divorced/Seperated', 'Dont know'])
        input_data['education_level'] = st.selectbox('Education Level', ['Secondary education', 'No formal education',
       'Vocational/Specialised training', 'Primary education',
       'Tertiary education', 'Other/Dont know/RTA'])
        input_data['job_type'] = st.selectbox('Job Type', ['Self employed', 'Government Dependent',
       'Formally employed Private', 'Informally employed',
       'Formally employed Government', 'Farming and Fishing',
       'Remittance Dependent', 'Other Income',
       'Dont Know/Refuse to answer', 'No Income'])
        input_data['gender_of_respondent'] = st.selectbox('Gender of Respondent', ['Female', 'Male'])

    input_df = pd.DataFrame([input_data])  # Convert collected data into a DataFrame
    st.write(input_df)  # Display the collected data on the app interface

    if st.button('Predict'):  # When the 'Predict' button is clicked
        final_df = preprocessor(input_df)  # Preprocess the collected data
        prediction = model.predict(final_df)[0]  # Use the model to predict the outcome
        
        # Display the prediction result
        if prediction == 1:
            st.write('There is high chance this person has  a bank account')
        else:
            st.write('There is high chance this person does not have a bank account')

    # Add file uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Preprocess the uploaded data
        preprocessed_data = preprocess_data(data)

        # Make predictions
        predictions = model.predict(preprocessed_data)

        st.write("Predictions:")
        st.write(predictions)

if __name__ == '__main__':
    main()

    
    
    
    
        