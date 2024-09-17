from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            age = float(request.form['age'])
            gender = float(request.form['gender'])
            heart_rate = float(request.form['heart_rate'])
            systolic_bp = float(request.form['systolic_bp'])
            diastolic_bp = float(request.form['diastolic_bp'])
            blood_sugar = float(request.form['blood_sugar'])
            ck_mb = float(request.form['ck_mb'])
            troponin = float(request.form['troponin'])

            # Create an array for prediction
            features = np.array([[age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin]])

            # Make a prediction using the trained model
            prediction = model.predict(features)

            # Convert the prediction to a readable result
            result = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'

            return render_template('predict.html', prediction_result=result)

        except ValueError:
            return "Please enter valid input values."

if __name__ == '__main__':
    app.run(debug=True)
