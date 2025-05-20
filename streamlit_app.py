import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Streamlit UI
st.set_page_config(page_title='Malaria Prediction Model', page_icon='🦟', layout='centered')
st.title('🦟 Malaria Prediction Model')
st.write("This is a robust Malaria prediction model using XGBoost, deployed with Streamlit.")

@st.cache_resource
def train_model():
    # Simulate a Larger and More Realistic Synthetic Dataset
    np.random.seed(42)
    data_size = 10000  # Larger dataset for better learning
    data = pd.DataFrame({
        'Fever': np.random.randint(0, 2, data_size),
        'Headache': np.random.randint(0, 2, data_size),
        'Muscle_Pain': np.random.randint(0, 2, data_size),
        'Nausea': np.random.randint(0, 2, data_size),
        'Fatigue': np.random.randint(0, 2, data_size)
    })

    # Create more variable Malaria conditions
    data['Malaria'] = (
        (data['Fever'] & data['Headache']) | 
        ((data['Fever'] | data['Nausea']) & (data['Fatigue'] | data['Muscle_Pain'])) | 
        ((data['Fever'] == 1) & (np.random.rand(data_size) > 0.3))
    ).astype(int)

    # Train the XGBoost model
    X = data.drop(columns=['Malaria'])
    y = data['Malaria']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
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
    probability = model.predict_proba(user_input)[0][1] * 100
    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts: You are likely to have Malaria. ({probability:.2f}% confidence)")
    else:
        st.success(f"✅ The model predicts: You are NOT likely to have Malaria. ({100 - probability:.2f}% confidence)")
