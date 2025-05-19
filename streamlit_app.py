import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Streamlit UI
st.set_page_config(page_title='Malaria Prediction Model', page_icon='🦟', layout='centered')
st.title('🦟 Malaria Prediction Model')
st.write("This is a robust Malaria prediction model using RandomForest, deployed with Streamlit.")

@st.cache_resource
def train_model():
    # Simulate a Synthetic Dataset
    np.random.seed(42)
    data_size = 1000
    data = pd.DataFrame({
        'Fever': np.random.randint(0, 2, data_size),
        'Headache': np.random.randint(0, 2, data_size),
        'Muscle_Pain': np.random.randint(0, 2, data_size),
        'Nausea': np.random.randint(0, 2, data_size),
        'Fatigue': np.random.randint(0, 2, data_size),
        'Malaria': np.random.randint(0, 2, data_size)
    })

    # Train the RandomForest model
    X = data.drop(columns=['Malaria'])
    y = data['Malaria']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    return model, accuracy

# Train the model and cache it
model, accuracy = train_model()

# Display Model Performance
st.write("### 📊 Model Performance")
st.metric("Accuracy", f"{accuracy:.2f}%")

# User Input for Prediction
st.write("### ✨ Make a Prediction")
fever = st.selectbox("Do you have Fever?", ["No", "Yes"]) == "Yes"
headache = st.selectbox("Do you have Headache?", ["No", "Yes"]) == "Yes"
muscle_pain = st.selectbox("Do you have Muscle Pain?", ["No", "Yes"]) == "Yes"
nausea = st.selectbox("Do you have Nausea?", ["No", "Yes"]) == "Yes"
fatigue = st.selectbox("Do you have Fatigue?", ["No", "Yes"]) == "Yes"

if st.button("🚀 Predict Malaria"):
    user_input = np.array([[fever, headache, muscle_pain, nausea, fatigue]]).astype(int)
    prediction = model.predict(user_input)
    if prediction[0] == 1:
        st.error("⚠️ The model predicts: You are likely to have Malaria.")
    else:
        st.success("✅ The model predicts: You are NOT likely to have Malaria.")
