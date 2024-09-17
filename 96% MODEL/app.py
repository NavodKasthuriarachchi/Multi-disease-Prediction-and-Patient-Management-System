from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('diabetes_rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Home route that renders the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        pregnancies = int(request.form['Pregnancies'])
        glucose = float(request.form['Glucose'])
        blood_pressure = float(request.form['BloodPressure'])
        skin_thickness = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
        age = float(request.form['Age'])
        
        # Prepare data for prediction
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
        
        # Scale the input data
        data = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(data)
        
        # Return the result
        if prediction[0] == 1:
            result = 'The patient is likely to have diabetes.'
        else:
            result = 'The patient is unlikely to have diabetes.'

        return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
