# app.py

import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define the path to the model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'titanic_rf_model.pkl')

# Load the trained Random Forest model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    print("Please ensure 'titanic_rf_model.pkl' is in a 'model' subdirectory.")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """
    Renders the main prediction form page.
    """
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                             prediction_text="❌ Error: Model not loaded. Please check server logs.")

    try:
        # Get data from the form
        pclass = int(request.form['pclass'])
        sex = request.form['sex']  # 'male' or 'female'
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']  # 'C', 'Q', 'S'

        # Data Preprocessing (MUST match model training)
        # Convert 'Sex' to numerical
        sex_encoded = 1 if sex == 'female' else 0

        # Handle 'Embarked' with one-hot encoding
        embarked_q = 1 if embarked == 'Q' else 0
        embarked_s = 1 if embarked == 'S' else 0

        # Create 'FamilySize' feature
        family_size = sibsp + parch + 1

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_q, embarked_s, family_size]],
                                  columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S', 'FamilySize'])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Interpret the prediction
        if prediction == 1:
            result = "🎉 Survived!"
            probability = prediction_proba[1] * 100
            emoji = "✅"
        else:
            result = "💔 Not Survived"
            probability = prediction_proba[0] * 100
            emoji = "❌"

        prediction_text = f"{emoji} Prediction: {result} (Confidence: {probability:.1f}%)"

        return render_template('index.html', prediction_text=prediction_text)

    except ValueError as e:
        return render_template('index.html', 
                             prediction_text="❌ Error: Please enter valid numerical values.")
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', 
                             prediction_text="❌ Error: Something went wrong during prediction.")

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html', 
                         prediction_text="❌ Page not found. Please check the URL."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('index.html', 
                         prediction_text="❌ Internal server error. Please try again."), 500

if __name__ == '__main__':
    # Environment-based configuration
    is_development = os.environ.get('FLASK_ENV') == 'development'
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("🚢 Starting Titanic Survival Predictor...")
    print(f"📊 Model status: {'✅ Loaded' if model else '❌ Not loaded'}")
    print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
    print(f"🌐 Environment: {'Development' if is_development else 'Production'}")
    
    if debug_mode and not is_development:
        print("⚠️  WARNING: Debug mode is ON in production environment!")
    
    app.run(
        debug=debug_mode,
        host='127.0.0.1' if is_development else '0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )