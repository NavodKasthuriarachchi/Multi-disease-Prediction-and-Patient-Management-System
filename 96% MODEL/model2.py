# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Handle missing values (replace zero values in columns with median values, except for 'Pregnancies' and 'Outcome')
columns_with_zeroes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
imputer = SimpleImputer(missing_values=0, strategy='median')
X[columns_with_zeroes] = imputer.fit_transform(X[columns_with_zeroes])

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy}")

# Save the trained model to a file
with open('diabetes_rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# Save the scaler to a file
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Flask application code starts here

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
        age = int(request.form['Age'])
        
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
