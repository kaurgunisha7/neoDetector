import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title and description
st.title('Nearest Earth Objects Hazard Prediction')
st.write('Predict whether a near-Earth object is hazardous based on its characteristics.')

# Input features
st.header('Input Features')

# Define input fields for each feature
def user_input_features():
    absolute_magnitude = st.number_input('Absolute Magnitude', min_value=0.0, max_value=40.0, value=22.0)
    estimated_diameter_min = st.number_input('Estimated Diameter Min (km)', min_value=0.0, max_value=10.0, value=0.1)
    estimated_diameter_max = st.number_input('Estimated Diameter Max (km)', min_value=0.0, max_value=10.0, value=0.3)
    relative_velocity = st.number_input('Relative Velocity (km/s)', min_value=0.0, max_value=1000000.0, value=20000.0)
    miss_distance = st.number_input('Miss Distance (LD)', min_value=0.0, max_value=10.580105e+07, value=1.580105e+07)
    orbiting_body = st.selectbox('Orbiting Body', ['Earth'])  # Adjust if there are other options

    data = {
        'absolute_magnitude': absolute_magnitude,
        'estimated_diameter_min': estimated_diameter_min,
        'estimated_diameter_max': estimated_diameter_max,
        'relative_velocity': relative_velocity,
        'miss_distance': miss_distance,
        # 'orbiting_body': orbiting_body  # Include if your model uses this feature
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocessing
# If your model requires scaling, apply the scaler
input_scaled = scaler.transform(input_df)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('The object is **Hazardous**.')
    else:
        st.write('The object is **Not Hazardous**.')

    st.subheader('Prediction Probability')
    st.write(f'Hazardous: {prediction_proba[0][1]*100:.2f}%')
    st.write(f'Not Hazardous: {prediction_proba[0][0]*100:.2f}%')

# Optional: Display data visualizations
st.header('Data Visualizations')

# Correlation Heatmap
if st.checkbox('Show Correlation Heatmap'):
    st.subheader('Correlation Heatmap')
    data = pd.read_csv('nearest-earth-objects(1910-2024).csv')
    data = data.drop(['name','neo_id','orbiting_body'],axis=1)
    data = data.dropna()
    data.drop_duplicates(inplace=True)
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)

# Feature Importance
if st.checkbox('Show Feature Importance'):
    st.subheader('Feature Importance')
    importances = model.feature_importances_
    feature_names = input_df.columns
    feat_importances = pd.Series(importances, index=feature_names)
    feat_importances.nlargest(10).plot(kind='barh')
    st.pyplot(plt)
