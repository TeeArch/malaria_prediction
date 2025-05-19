import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit UI
st.set_page_config(page_title='Malaria Prediction Model', page_icon='🦟', layout='centered')
st.title('🦟 Malaria Prediction Model')
st.write("This is a simple Malaria prediction model deployed with Streamlit.")

# Simulate a Synthetic Dataset
np.random.seed(42)
data_size = 1000

# Generate random data for health features
data = pd.DataFrame({
    'Fever': np.random.randint(0, 2, data_size),
    'Headache': np.random.randint(0, 2, data_size),
    'Muscle_Pain': np.random.randint(0, 2, data_size),
    'Nausea': np.random.randint(0, 2, data_size),
    'Fatigue': np.random.randint(0, 2, data_size),
    'Malaria': np.random.randint(0, 2, data_size)
})

st.write("### 📊 Generated Health Data Sample:")
st.dataframe(data.head(), use_container_width=True)

# Split the data
X = data.drop(columns=['Malaria'])
y = data['Malaria']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Model Performance
predictions = model.predict(X_test)
st.write("### 📊 Model Performance")
st.metric("Accuracy", round(accuracy_score(y_test, predictions), 4))

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
