from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('dangerous_disease_model.h5')
scaler = joblib.load('scaler.pkl')

# Feature names from the training dataset
feature_names = scaler.feature_names_in_

# Define symptoms to query the user
symptoms_to_ask = [
    'Fever', 'Diarrhea', 'Vomiting', 'Weight loss', 'Dehydration', 
    'Coughing', 'Tiredness', 'Pains', 'Difficulty breathing', 'Lethargy'
]

# Map the symptoms to their corresponding feature names in the dataset
symptom_to_feature_map = {symptom: f"symptoms{i}_{symptom}" for i, symptom in enumerate(symptoms_to_ask)}

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_to_ask)

@app.route('/predict', methods=['POST'])
def predict():
    # Initialize features
    user_symptoms = {feature: 0 for feature in feature_names}
    
    # Gather user input
    for symptom, feature in symptom_to_feature_map.items():
        if request.form.get(symptom) == 'Yes' and feature in user_symptoms:
            user_symptoms[feature] = 1
    
    # Convert to DataFrame
    user_symptoms_df = pd.DataFrame([user_symptoms])[feature_names]
    
    # Scale features
    symptoms_scaled = scaler.transform(user_symptoms_df)
    
    # Predict using the model
    prediction = model.predict(symptoms_scaled)
    
    # Interpret prediction
    result = "The disease is dangerous!" if prediction >= 0.5 else "The disease is not dangerous."
    return render_template('index.html', symptoms=symptoms_to_ask, result=result)

if __name__ == "__main__":
    app.run(debug=True)
