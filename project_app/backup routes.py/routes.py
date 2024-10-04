from flask import render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user, login_required
from project_app import app# type: ignore
from project_app.__init__ import mysql# type: ignore
from project_app.models import User # type: ignore
import os
import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from flask import send_from_directory
from MySQLdb.cursors import DictCursor
from flask import Flask, render_template, request, redirect, url_for, flash, current_app, jsonify
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template

# Home route
@app.route('/')
# @login_required
def home():
    return render_template('home.html')

# Load the diabetes prediction model.....................................................................................................................................................................................................................
with open("J:/Final Project/multi_disease_system/project_app/diabetes.pkl", 'rb') as model_file:
    model = pickle.load(model_file)
with open("J:/Final Project/For edit/multi_disease_system/project_app/scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Route to display the prediction page
# Route to display the prediction page
@app.route('/predict', methods=['GET', 'POST'])
@login_required  # Ensure that only logged-in users can access this
def add_prediction():
    cur = mysql.connection.cursor()
    cur.execute("SELECT id, first_name, last_name FROM patients")
    patients = cur.fetchall()

    if request.method == 'POST':
        try:
            # Retrieve form data from the request
            patient_id = request.form.get('patient_id')
            pregnancies = float(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            blood_pressure = float(request.form['BloodPressure'])
            skin_thickness = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            diabetes_pedigree = float(request.form['DiabetesPedigreeFunction'])
            age = float(request.form['Age'])

            # Get the actual nurse_id from the nurses table based on the current user's id
            cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
            nurse_data = cur.fetchone()

            if not nurse_data:
                flash('Error: Nurse record not found for the current user.')
                return redirect(url_for('add_prediction'))

            nurse_id = nurse_data[0]  # Extract the nurse_id

            # Prepare data for prediction
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])

            # Scale the input data
            input_data = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data)

            # Interpret the prediction result
            prediction_result = "There is a risk of developing diabetes Diabetes" if prediction[0] == 1 else "There is NO risk of Diabetes"

            # Debugging: Print values to ensure all variables are correct
            print(f"patient_id: {patient_id}, nurse_id: {nurse_id}, pregnancies: {pregnancies}, glucose: {glucose}, blood_pressure: {blood_pressure}")
            print(f"skin_thickness: {skin_thickness}, insulin: {insulin}, bmi: {bmi}, diabetes_pedigree: {diabetes_pedigree}, age: {age}, prediction_result: {prediction_result}")

            # Insert prediction into the 'predictions' table
            query = """
                INSERT INTO predictions 
                (patient_id, nurse_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, prediction_result) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.execute(query, (patient_id, nurse_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, prediction_result))

            # Commit the changes to the database
            mysql.connection.commit()
            cur.close()

            # Flash a message and return the template with the prediction result
            flash('Prediction made and stored successfully!')
            return render_template('index.html', prediction_text=f"The prediction is {prediction_result}.", patients=patients)

        except Exception as e:
            flash(f"Error in making prediction: {str(e)}")
            return redirect(url_for('add_prediction'))

    # Render the prediction form
    return render_template('index.html', patients=patients)


#predict heart disease...................................................................................................................................................................................................................................

# Load the heart disease model (assuming the path is correct)
heart_model_path = 'J:/Final Project/For edit/multi_disease_system/project_app/heart_disease_model.pkl'
heart_model = joblib.load(heart_model_path)

# Route to display the heart disease prediction page
@app.route('/predict_heart', methods=['GET', 'POST'])
@login_required  # Ensure only logged-in users can access
def predict_heart_disease():
    # Check if the current user is a nurse
    if current_user.role != 'nurse':
        flash('Unauthorized access! Only nurses can make predictions.')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()
    cur.execute("SELECT id, first_name, last_name FROM patients")
    patients = cur.fetchall()

    if request.method == 'POST':
        try:
            # Retrieve form data
            patient_id = request.form.get('patient_id')
            age = float(request.form['age'])
            gender = float(request.form['gender'])
            heart_rate = float(request.form['heart_rate'])
            systolic_bp = float(request.form['systolic_bp'])
            diastolic_bp = float(request.form['diastolic_bp'])
            blood_sugar = float(request.form['blood_sugar'])
            ck_mb = float(request.form['ck_mb'])
            troponin = float(request.form['troponin'])

            # Get the actual nurse_id from the nurses table based on the current user's id
            cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
            nurse_data = cur.fetchone()

            if not nurse_data:
                flash('Error: Nurse record not found for the current user.')
                return redirect(url_for('predict_heart_disease'))

            nurse_id = nurse_data[0]  # Extract the nurse_id

            # Prepare the data for prediction
            input_data = np.array([[age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin]])

            # Make prediction using the heart disease model
            prediction = heart_model.predict(input_data)
            prediction_result = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'

            # Insert the prediction into the 'heart_predictions' table
            query = """
                INSERT INTO heart_predictions 
                (patient_id, nurse_id, age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin, prediction_result)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.execute(query, (patient_id, nurse_id, age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin, prediction_result))
            mysql.connection.commit()
            cur.close()

            # Flash a message and return the template with the prediction result
            flash('Heart Disease prediction made and stored successfully!')
            return render_template('predict_heart.html', prediction_result=prediction_result, patients=patients)

        except Exception as e:
            flash(f"Error in making prediction: {str(e)}")
            return redirect(url_for('predict_heart_disease'))

    return render_template('predict_heart.html', patients=patients)



#Register Nurse...................................................................................................................................................................................................................................
# @app.route('/register_nurse', methods=['GET', 'POST'])
# def register_nurse():
#     if request.method == 'POST':
#         # Get form data
#         username = request.form['username']
#         password = request.form['password']
#         email = request.form['email']
#         first_name = request.form['first_name']
#         last_name = request.form['last_name']
#         contact_number = request.form['contact_number']
        
#         # Insert user into `users` table
#         cur = mysql.connection.cursor()
#         cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'nurse')", 
#                     (username, password, email))
#         user_id = cur.lastrowid  # Get the user_id of the inserted user
        
#         # Insert nurse details into `nurses` table
#         cur.execute("INSERT INTO nurses (user_id, first_name, last_name, contact_number) VALUES (%s, %s, %s, %s)", 
#                     (user_id, first_name, last_name, contact_number))
#         mysql.connection.commit()
#         cur.close()

#         flash('Nurse registered successfully!')
#         return redirect(url_for('login'))

#     return render_template('register_nurse.html')
@app.route('/register_nurse', methods=['GET', 'POST'])
def register_nurse():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']  # User's input password
        hashed_password = generate_password_hash(password)  # Hash the password
        email = request.form['email']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        contact_number = request.form['contact_number']
        
        # Insert user into `users` table with hashed password
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'nurse')", 
                    (username, hashed_password, email))  # Store the hashed password
        user_id = cur.lastrowid  # Get the user_id of the inserted user
        
        # Insert nurse details into `nurses` table
        cur.execute("INSERT INTO nurses (user_id, first_name, last_name, contact_number) VALUES (%s, %s, %s, %s)", 
                    (user_id, first_name, last_name, contact_number))
        mysql.connection.commit()
        cur.close()

        flash('Nurse registered successfully!')
        return redirect(url_for('login'))

    return render_template('register_nurse.html')
#Register Doctor...................................................................................................................................................................................................................................
# @app.route('/register_doctor', methods=['GET', 'POST'])
# def register_doctor():
#     if request.method == 'POST':
#         # Get form data
#         username = request.form['username']
#         password = request.form['password']
#         email = request.form['email']
#         first_name = request.form['first_name']
#         last_name = request.form['last_name']
#         specialization = request.form['specialization']
#         contact_number = request.form['contact_number']
        
#         # Insert user into `users` table
#         cur = mysql.connection.cursor()
#         cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'doctor')", 
#                     (username, password, email))
#         user_id = cur.lastrowid
        
#         # Insert doctor details into `doctors` table
#         cur.execute("INSERT INTO doctors (user_id, first_name, last_name, specialization, contact_number) VALUES (%s, %s, %s, %s, %s)", 
#                     (user_id, first_name, last_name, specialization, contact_number))
#         mysql.connection.commit()
#         cur.close()

#         flash('Doctor registered successfully!')
#         return redirect(url_for('login'))

#     return render_template('register_doctor.html')

from werkzeug.security import generate_password_hash
import re

@app.route('/register_doctor', methods=['GET', 'POST'])
def register_doctor():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        specialization = request.form['specialization']
        contact_number = request.form['contact_number']

        # Validations

        # Check if any required field is missing
        if not username or not password or not email or not first_name or not last_name or not specialization or not contact_number:
            flash('All fields are required!')
            return redirect(url_for('register_doctor'))

        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash('Invalid email format!')
            return redirect(url_for('register_doctor'))

        # Validate password length and complexity (at least 8 characters, including a number and a special character)
        if len(password) < 8 or not re.search(r"\d", password) or not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            flash('Password must be at least 8 characters long and include a number and a special character!')
            return redirect(url_for('register_doctor'))

        # Check if username already exists
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        existing_user = cur.fetchone()
        if existing_user:
            flash('Username already exists! Please choose another one.')
            return redirect(url_for('register_doctor'))

        # Hash the password before storing it
        hashed_password = generate_password_hash(password)

        # Insert user into `users` table
        cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'doctor')", 
                    (username, hashed_password, email))
        user_id = cur.lastrowid
        
        # Insert doctor details into `doctors` table
        cur.execute("INSERT INTO doctors (user_id, first_name, last_name, specialization, contact_number) VALUES (%s, %s, %s, %s, %s)", 
                    (user_id, first_name, last_name, specialization, contact_number))
        mysql.connection.commit()
        cur.close()

        flash('Doctor registered successfully!')
        return redirect(url_for('login'))

    return render_template('register_doctor.html')

#Register patient...................................................................................................................................................................................................................................
@app.route('/register_patient', methods=['POST'])
@login_required
def register_patient():
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    # Get form data for the patient
    first_name = request.form['first_name']
    last_name = request.form['last_name']
    age = request.form['age']
    gender = request.form['gender']
    contact_number = request.form['contact_number']
    username = request.form['username']  # Add a username field for patient registration
    password = request.form['password']  # Add a password field for patient registration
    email = request.form['email']  # Add an email field for patient registration

    # Insert the new patient into the users table first, with role as 'patient'
    cur = mysql.connection.cursor()
    cur.execute(
        "INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'patient')",
        (username, password, email)
    )
    mysql.connection.commit()

    # Get the user_id of the newly inserted patient
    user_id = cur.lastrowid

    # Fetch the nurse's id from the nurses table using the current user's id
    cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
    nurse_data = cur.fetchone()

    if not nurse_data:
        flash('Nurse not found!')
        return redirect(url_for('nurse_dashboard'))

    nurse_id = nurse_data[0]  # Extract the nurse's id

    # Insert the new patient into the patients table, linking with the user_id and nurse_id
    cur.execute(
        "INSERT INTO patients (user_id, first_name, last_name, age, gender, contact_number, nurse_id) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (user_id, first_name, last_name, age, gender, contact_number, nurse_id)
    )
    mysql.connection.commit()
    cur.close()

    flash('Patient registered successfully!')
    return redirect(url_for('nurse_dashboard'))

# from werkzeug.security import generate_password_hash

# @app.route('/register_patient', methods=['POST'])
# @login_required
# def register_patient():
#     if current_user.role != 'nurse':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     # Get form data for the patient
#     first_name = request.form['first_name']
#     last_name = request.form['last_name']
#     age = request.form['age']
#     gender = request.form['gender']
#     contact_number = request.form['contact_number']
#     username = request.form['username']  # Add a username field for patient registration
#     password = request.form['password']  # Add a password field for patient registration
#     email = request.form['email']  # Add an email field for patient registration

#     # Hash the patient's password before storing it
#     hashed_password = generate_password_hash(password)

#     # Insert the new patient into the users table first, with role as 'patient'
#     cur = mysql.connection.cursor()
#     cur.execute(
#         "INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'patient')",
#         (username, hashed_password, email)  # Store the hashed password
#     )
#     mysql.connection.commit()

#     # Get the user_id of the newly inserted patient
#     user_id = cur.lastrowid

#     # Fetch the nurse's id from the nurses table using the current user's id
#     cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
#     nurse_data = cur.fetchone()

#     if not nurse_data:
#         flash('Nurse not found!')
#         return redirect(url_for('nurse_dashboard'))

#     nurse_id = nurse_data[0]  # Extract the nurse's id

#     # Insert the new patient into the patients table, linking with the user_id and nurse_id
#     cur.execute(
#         "INSERT INTO patients (user_id, first_name, last_name, age, gender, contact_number, nurse_id) "
#         "VALUES (%s, %s, %s, %s, %s, %s, %s)",
#         (user_id, first_name, last_name, age, gender, contact_number, nurse_id)
#     )
#     mysql.connection.commit()
#     cur.close()

#     flash('Patient registered successfully!')
#     return redirect(url_for('nurse_dashboard'))


    # if request.method == 'POST':
    #     # Get form data
    #     username = request.form['username']
    #     password = request.form['password']
    #     email = request.form['email']
    #     first_name = request.form['first_name']
    #     last_name = request.form['last_name']
    #     age = request.form['age']
    #     gender = request.form['gender']
    #     contact_number = request.form['contact_number']
        
    #     # Insert user into `users` table
    #     cur = mysql.connection.cursor()
    #     cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'patient')", 
    #                 (username, password, email))
    #     user_id = cur.lastrowid
        
    #     # Insert patient details into `patients` table
    #     cur.execute("INSERT INTO patients (user_id, first_name, last_name, age, gender, contact_number) VALUES (%s, %s, %s, %s, %s, %s)", 
    #                 (user_id, first_name, last_name, age, gender, contact_number))
    #     mysql.connection.commit()
    #     cur.close()

    #     flash('Patient registered successfully!')
    #     return redirect(url_for('login'))

    # return render_template('register_patient.html')
# Login route...................................................................................................................................................................................................................................
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user_data = cur.fetchone()
        


        print(f"User Data: {user_data}")  # Check what is being retrieved from the database 

        # Assuming the database order is: id, username, password, email, role, created_at
        if user_data and user_data[2] == password:  # Basic password check, consider using hashed passwords in production
            # Pass only id, username, password, and role to the User constructor (ignore created_at)
            user = User(user_data[0], user_data[1], user_data[2], user_data[4])
            login_user(user)

            print(f"Logged in as {user.username}, Role: {user.role}")
            
            # After login, check the role from current_user object
            if user.role == 'patient':
                return redirect(url_for('patient_dashboard'))
            elif user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif user.role == 'nurse':
                return redirect(url_for('nurse_dashboard'))
            elif user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            
            else:
                 flash(f"Unknown role: {user.role}!")
                 return redirect(url_for('login'))

        else:
            flash('Invalid credentials')
            return redirect(url_for('login'))

    return render_template('login.html')


# Logout route
# @app.route('/logout')
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('login'))

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
        
#         # Fetch user data from the database
#         cur = mysql.connection.cursor()
#         cur.execute("SELECT * FROM users WHERE username = %s", (username,))
#         user_data = cur.fetchone()
#         cur.close()

#         # Debugging: Print the user data fetched from the database
#         print(f"User Data: {user_data}")

#         # Assuming the database order is: id, username, password (hashed), email, role, created_at
#         if user_data:
#             # Debugging: Print the role and password data
#             print(f"Stored Hashed Password: {user_data[2]}")
#             print(f"Role: {user_data[4]}")

#             # Verify the password using check_password_hash
#             if check_password_hash(user_data[2], password):
#                 # Pass only id, username, and role to the User constructor (ignore created_at)
#                 user = User(user_data[0], user_data[1], user_data[2], user_data[4])
#                 login_user(user)

#                 print(f"Logged in as {user.username}, Role: {user.role}")

#                 # After login, check the role from current_user object and redirect accordingly
#                 if user.role == 'patient':
#                     return redirect(url_for('patient_dashboard'))
#                 elif user.role == 'doctor':
#                     return redirect(url_for('doctor_dashboard'))
#                 elif user.role == 'nurse':
#                     return redirect(url_for('nurse_dashboard'))
#                 elif user.role == 'admin':
#                     return redirect(url_for('admin_dashboard'))
#                 else:
#                     flash(f"Unknown role: {user.role}!")
#                     return redirect(url_for('login'))

#             else:
#                 flash('Invalid password')
#                 print("Password check failed")  # Debugging: Password verification failed
#                 return redirect(url_for('login'))
#         else:
#             flash('Invalid username')
#             print("Username not found")  # Debugging: No user found for this username
#             return redirect(url_for('login'))

#     return render_template('login.html')


# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# .................................................................................................................................Load your prediction model (e.g., for diabetes)
# model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Nurse Dashboard..............................................................................Original.....................................................................................................................................................
# @app.route('/nurse_dashboard')
# @login_required
# def nurse_dashboard():
#     if current_user.role != 'nurse':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     # Fetch the nurse's id from the nurses table using the current user's id
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
#     nurse_data = cur.fetchone()

#     if not nurse_data:
#         flash('Nurse not found!')
#         return redirect(url_for('home'))

#     nurse_id = nurse_data[0]

#     # Fetch the patients registered by this nurse
#     cur.execute("SELECT * FROM patients WHERE nurse_id = %s", (nurse_id,))
#     patients = cur.fetchall()

#     # Fetch the list of doctors
#     cur.execute("SELECT id, first_name, last_name FROM doctors")
#     doctors = cur.fetchall()


#     # Fetch the appointments made by the nurse's patients
#     cur.execute("""
#     SELECT 
#         a.id, a.appointment_date, a.appointment_time, a.reason, 
#         p.first_name, p.last_name, d.first_name, d.last_name
#     FROM appointments a
#     JOIN patients p ON a.patient_id = p.id
#     JOIN doctors d ON a.doctor_id = d.id
#     WHERE p.nurse_id = %s
#     ORDER BY a.appointment_date, a.appointment_time
#     """, (nurse_id,))
#     appointments = cur.fetchall()


#     # Pass both patients and appointments to the template
#     print(appointments)
#     return render_template('nurse_dashboard.html', patients=patients, appointments=appointments, doctors=doctors)

#Edited Nurse Dashboard.......................................................................................................................*.*.*...............................*.*.*................................................

# Nurse Dashboard...................................................................................................................................................................................................................................
@app.route('/nurse_dashboard')
@login_required
def nurse_dashboard():
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    # Fetch the nurse's id from the nurses table using the current user's id
    cur = mysql.connection.cursor()
    cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
    nurse_data = cur.fetchone()

    if not nurse_data:
        flash('Nurse not found!')
        return redirect(url_for('home'))

    nurse_id = nurse_data[0]

    # Fetch the patients registered by this nurse
    cur.execute("SELECT * FROM patients WHERE nurse_id = %s", (nurse_id,))
    patients = cur.fetchall()

    # Fetch the list of doctors
    cur.execute("SELECT id, first_name, last_name FROM doctors")
    doctors = cur.fetchall()


    # Fetch the appointments made by the nurse's patients
    cur.execute("""
    SELECT 
        a.id, a.appointment_date, a.appointment_time, a.reason, 
        p.first_name, p.last_name, d.first_name, d.last_name
    FROM appointments a
    JOIN patients p ON a.patient_id = p.id
    JOIN doctors d ON a.doctor_id = d.id
    WHERE p.nurse_id = %s
    ORDER BY a.appointment_date, a.appointment_time
    """, (nurse_id,))
    appointments = cur.fetchall()


    # Pass both patients and appointments to the template
    print(appointments)
    return render_template('nurse_dashboard.html', patients=patients, appointments=appointments, doctors=doctors)
# Doctor Dashboard...................................................................................................................................................................................................................................

@app.route('/doctor_dashboard', methods=['GET', 'POST'])
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Fetch the doctor's appointments along with patient details
    cur.execute("""
        SELECT 
            a.appointment_date, 
            a.appointment_time, 
            a.reason, 
            p.first_name, 
            p.last_name, 
            p.age, 
            p.contact_number,
            p.id  -- Ensure patient ID is fetched here
        FROM appointments a
        JOIN patients p ON a.patient_id = p.id
        WHERE a.doctor_id = (SELECT id FROM doctors WHERE user_id = %s)
        ORDER BY a.appointment_date, a.appointment_time
    """, (current_user.id,))
    
    appointments = cur.fetchall()

    # Fetch the prediction data for the patients
    cur.execute("""
        SELECT 
            p.first_name AS patient_first_name, 
            p.last_name AS patient_last_name, 
            pr.pregnancies, 
            pr.glucose, 
            pr.blood_pressure, 
            pr.skin_thickness, 
            pr.insulin, 
            pr.bmi, 
            pr.diabetes_pedigree, 
            pr.age, 
            pr.prediction_result,
            n.first_name AS nurse_first_name, 
            n.last_name AS nurse_last_name
        FROM predictions pr
        JOIN patients p ON pr.patient_id = p.id
        JOIN nurses n ON pr.nurse_id = n.id
    """)
    
    predictions = cur.fetchall()

    # Fetch the heart disease prediction data for the patients
    cur.execute("""
        SELECT 
            p.first_name AS patient_first_name, 
            p.last_name AS patient_last_name, 
            hp.age, 
            hp.gender, 
            hp.heart_rate, 
            hp.systolic_bp, 
            hp.diastolic_bp, 
            hp.blood_sugar, 
            hp.ck_mb, 
            hp.troponin, 
            hp.prediction_result,
            n.first_name AS nurse_first_name, 
            n.last_name AS nurse_last_name
        FROM heart_predictions hp
        JOIN patients p ON hp.patient_id = p.id
        JOIN nurses n ON hp.nurse_id = n.id
    """)
    heart_disease_predictions = cur.fetchall()


    cur.close()

    return render_template('doctor_dashboard.html', appointments=appointments, predictions=predictions, heart_disease_predictions=heart_disease_predictions)

# Doctor Dashboard - View Appointments, Predictions, and Upload Reports.....................................................................................................*.*.*.................................
# @app.route('/doctor_dashboard', methods=['GET', 'POST'])
# @login_required
# def doctor_dashboard():
#     if current_user.role != 'doctor':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     cur = mysql.connection.cursor()

#     # Fetch the doctor's appointments along with patient details
#     cur.execute("""
#         SELECT 
#             a.appointment_date, 
#             a.appointment_time, 
#             a.reason, 
#             p.first_name, 
#             p.last_name, 
#             p.age, 
#             p.contact_number,
#             p.id AS patient_id  -- Ensure patient ID is fetched here
#         FROM appointments a
#         JOIN patients p ON a.patient_id = p.id
#         WHERE a.doctor_id = (SELECT id FROM doctors WHERE user_id = %s)
#         ORDER BY a.appointment_date, a.appointment_time
#     """, (current_user.id,))
    
#     appointments = cur.fetchall()

#     # Fetch the prediction data for the patients
#     cur.execute("""
#         SELECT 
#             p.first_name AS patient_first_name, 
#             p.last_name AS patient_last_name, 
#             pr.pregnancies, 
#             pr.glucose, 
#             pr.blood_pressure, 
#             pr.skin_thickness, 
#             pr.insulin, 
#             pr.bmi, 
#             pr.diabetes_pedigree, 
#             pr.age, 
#             pr.prediction_result,
#             n.first_name AS nurse_first_name, 
#             n.last_name AS nurse_last_name
#         FROM predictions pr
#         JOIN patients p ON pr.patient_id = p.id
#         JOIN nurses n ON pr.nurse_id = n.id
#     """)
    
#     predictions = cur.fetchall()

#     # Fetch the heart disease prediction data for the patients
#     cur.execute("""
#         SELECT 
#             p.first_name AS patient_first_name, 
#             p.last_name AS patient_last_name, 
#             hp.age, 
#             hp.gender, 
#             hp.heart_rate, 
#             hp.systolic_bp, 
#             hp.diastolic_bp, 
#             hp.blood_sugar, 
#             hp.ck_mb, 
#             hp.troponin, 
#             hp.prediction_result,
#             n.first_name AS nurse_first_name, 
#             n.last_name AS nurse_last_name
#         FROM heart_predictions hp
#         JOIN patients p ON hp.patient_id = p.id
#         JOIN nurses n ON hp.nurse_id = n.id
#     """)
#     heart_disease_predictions = cur.fetchall()

#     # # Handle Report Upload Request
#     # if request.method == 'POST':
#     #     patient_id = request.form.get('patient_id')
#     #     report_name = request.form.get('report_name')
#     #     file = request.files.get('report_file')

#     #     # Debug the value of patient_id
#     #     print(f"Debug: patient_id = {patient_id}")

#     #     if not patient_id or not patient_id.isdigit():
#     #         flash('Invalid patient ID provided.')
#     #         return redirect(url_for('doctor_dashboard'))

#     #     if file and allowed_file(file.filename):
#     #         filename = secure_filename(file.filename)
#     #         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     #         file.save(file_path)

#     #         # Insert the report into the reports table
#     #         cur.execute("""
#     #             INSERT INTO reports (report_name, file_path, patient_id, doctor_id) 
#     #             VALUES (%s, %s, %s, (SELECT id FROM doctors WHERE user_id = %s))
#     #         """, (report_name, file_path, patient_id, current_user.id))
#     #         mysql.connection.commit()

#     #         flash('Report uploaded successfully!')
#     #         return redirect(url_for('doctor_dashboard'))
#     #     else:
#     #         flash('Invalid file type. Please upload a PDF or an image.')

#     cur.close()

#     return render_template('doctor_dashboard.html', appointments=appointments, predictions=predictions, heart_disease_predictions=heart_disease_predictions)





#Download Report...................................................................................................................................................................................................................................




from MySQLdb.cursors import DictCursor
#Patient Dashboard...................................................................................................................................................................................................................................
# @app.route('/patient_dashboard')
# @login_required
# def patient_dashboard():
#     # Ensure the user is a patient
#     if current_user.role != 'patient':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     # Use DictCursor to return results as dictionaries
#     cur = mysql.connection.cursor(DictCursor)

#     # Fetch patient ID based on the logged-in user
#     cur.execute("""
#         SELECT id FROM patients WHERE user_id = %s
#     """, (current_user.id,))
#     patient = cur.fetchone()

#     if not patient:
#         flash('Patient record not found!')
#         return redirect(url_for('home'))

#     patient_id = patient['id']  # Now you can access it like a dictionary

#     # # Fetch patient's prescriptions
#     # cur.execute("""
#     #     SELECT r.diagnosis, r.prescription, r.created_at, d.first_name AS doctor_first_name, d.last_name AS doctor_last_name 
#     #     FROM reports r
#     #     JOIN doctors d ON r.doctor_id = d.id
#     #     WHERE r.patient_id = %s
#     #     ORDER BY r.created_at DESC
#     # """, (patient_id,))
#     # prescriptions = cur.fetchall()

#     # Fetch patient's appointments
#     cur.execute("""
#         SELECT 
#             d.first_name, 
#             d.last_name, 
#             a.appointment_date, 
#             a.appointment_time, 
#             a.reason
#         FROM appointments a
#         JOIN doctors d ON a.doctor_id = d.id
#         WHERE a.patient_id = %s
#         ORDER BY a.appointment_date, a.appointment_time
#     """, (patient_id,))
#     appointments = cur.fetchall()

#     return render_template('patient_dashboard.html', appointments=appointments)


#patient Dashboard newly created.....................................................................................................................................................................................................................
# Patient Dashboard - View Appointments and Reports
# Patient Dashboard - View Appointments, Reports, and Predictions
@app.route('/patient_dashboard')
@login_required
def patient_dashboard():
    # Ensure the user is a patient
    if current_user.role != 'patient':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    # Use DictCursor to return results as dictionaries
    cur = mysql.connection.cursor(DictCursor)

    # Fetch patient ID based on the logged-in user
    cur.execute("""
        SELECT id FROM patients WHERE user_id = %s
    """, (current_user.id,))
    patient = cur.fetchone()

    if not patient:
        flash('Patient record not found!')
        return redirect(url_for('home'))

    patient_id = patient['id']  # Now you can access it like a dictionary

    # Fetch patient's reports
    cur.execute("""
        SELECT r.id, r.report_name, r.file_path, r.created_at, d.first_name AS doctor_first_name, d.last_name AS doctor_last_name 
        FROM reports r
        JOIN doctors d ON r.doctor_id = d.id
        WHERE r.patient_id = %s
        ORDER BY r.created_at DESC
    """, (patient_id,))
    reports = cur.fetchall()

    # Fetch patient's appointments
    cur.execute("""
        SELECT 
            d.first_name, 
            d.last_name, 
            a.appointment_date, 
            a.appointment_time, 
            a.reason
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.id
        WHERE a.patient_id = %s
        ORDER BY a.appointment_date, a.appointment_time
    """, (patient_id,))
    appointments = cur.fetchall()

    # Fetch patient's diabetes predictions
    cur.execute("""
        SELECT 
            p.pregnancies, p.glucose, p.blood_pressure, p.skin_thickness, p.insulin, 
            p.bmi, p.diabetes_pedigree, p.age, p.prediction_result, p.predicted_at,
            n.first_name AS nurse_first_name, n.last_name AS nurse_last_name
        FROM predictions p
        JOIN nurses n ON p.nurse_id = n.id
        WHERE p.patient_id = %s
    """, (patient_id,))
    diabetes_predictions = cur.fetchall()

    # Fetch patient's heart disease predictions
    cur.execute("""
        SELECT 
            hp.age, hp.gender, hp.heart_rate, hp.systolic_bp, hp.diastolic_bp, hp.blood_sugar, 
            hp.ck_mb, hp.troponin, hp.prediction_result, hp.predicted_at,
            n.first_name AS nurse_first_name, n.last_name AS nurse_last_name
        FROM heart_predictions hp
        JOIN nurses n ON hp.nurse_id = n.id
        WHERE hp.patient_id = %s
    """, (patient_id,))
    heart_predictions = cur.fetchall()

    return render_template('patient_dashboard.html', reports=reports, appointments=appointments, diabetes_predictions=diabetes_predictions, heart_predictions=heart_predictions)




# Download Report
@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    if current_user.role != 'patient':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    # Fetch report details to get the file path
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT file_path 
        FROM reports 
        WHERE id = %s AND patient_id = (SELECT id FROM patients WHERE user_id = %s)
    """, (report_id, current_user.id))
    report = cur.fetchone()

    if not report:
        flash('Report not found or unauthorized access!')
        return redirect(url_for('patient_dashboard'))

    file_path = report[0]
    if not os.path.exists(file_path):
        flash('The requested file does not exist.')
        return redirect(url_for('patient_dashboard'))

    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    try:
        return send_from_directory(directory, filename, as_attachment=True)
    except Exception as e:
        flash(f'Error in downloading the report: {str(e)}')
        return redirect(url_for('patient_dashboard'))


# Search Reports
# Search Reports
@app.route('/search_reports', methods=['GET', 'POST'])
@login_required
def search_reports():
    if current_user.role != 'patient':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    query = request.form.get('query', '').strip()

    # Use DictCursor to return results as dictionaries
    cur = mysql.connection.cursor(DictCursor)

    # Fetch patient ID based on the logged-in user
    cur.execute("""
        SELECT id FROM patients WHERE user_id = %s
    """, (current_user.id,))
    patient = cur.fetchone()

    if not patient:
        flash('Patient record not found!')
        return redirect(url_for('home'))

    patient_id = patient['id']

    if query:
        # Fetch reports that match the search query
        cur.execute("""
            SELECT r.id, r.report_name, r.file_path, r.created_at, d.first_name AS doctor_first_name, d.last_name AS doctor_last_name 
            FROM reports r
            JOIN doctors d ON r.doctor_id = d.id
            WHERE r.patient_id = %s AND r.report_name LIKE %s
            ORDER BY r.created_at DESC
        """, (patient_id, f"%{query}%"))
        reports = cur.fetchall()
    else:
        # If no query, fetch all reports
        cur.execute("""
            SELECT r.id, r.report_name, r.file_path, r.created_at, d.first_name AS doctor_first_name, d.last_name AS doctor_last_name 
            FROM reports r
            JOIN doctors d ON r.doctor_id = d.id
            WHERE r.patient_id = %s
            ORDER BY r.created_at DESC
        """, (patient_id,))
        reports = cur.fetchall()

    # Fetch all appointments (unfiltered)
    cur.execute("""
        SELECT 
            d.first_name, 
            d.last_name, 
            a.appointment_date, 
            a.appointment_time, 
            a.reason
        FROM appointments a
        JOIN doctors d ON a.doctor_id = d.id
        WHERE a.patient_id = %s
        ORDER BY a.appointment_date, a.appointment_time
    """, (patient_id,))
    appointments = cur.fetchall()

    return render_template('patient_dashboard.html', reports=reports, appointments=appointments, report_search_query=query, appointment_search_query=None)

# Search Appointments
@app.route('/search_appointments', methods=['GET', 'POST'])
@login_required
def search_appointments():
    if current_user.role != 'patient':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    query = request.form.get('query', '').strip()

    # Use DictCursor to return results as dictionaries
    cur = mysql.connection.cursor(DictCursor)

    # Fetch patient ID based on the logged-in user
    cur.execute("""
        SELECT id FROM patients WHERE user_id = %s
    """, (current_user.id,))
    patient = cur.fetchone()

    if not patient:
        flash('Patient record not found!')
        return redirect(url_for('home'))

    patient_id = patient['id']

    if query:
        # Fetch appointments that match the search query
        cur.execute("""
            SELECT 
                d.first_name, 
                d.last_name, 
                a.appointment_date, 
                a.appointment_time, 
                a.reason
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.patient_id = %s AND (d.first_name LIKE %s OR d.last_name LIKE %s OR a.reason LIKE %s)
            ORDER BY a.appointment_date, a.appointment_time
        """, (patient_id, f"%{query}%", f"%{query}%", f"%{query}%"))
        appointments = cur.fetchall()
    else:
        # If no query, fetch all appointments
        cur.execute("""
            SELECT 
                d.first_name, 
                d.last_name, 
                a.appointment_date, 
                a.appointment_time, 
                a.reason
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.id
            WHERE a.patient_id = %s
            ORDER BY a.appointment_date, a.appointment_time
        """, (patient_id,))
        appointments = cur.fetchall()

    # Fetch all reports (unfiltered)
    cur.execute("""
        SELECT r.id, r.report_name, r.file_path, r.created_at, d.first_name AS doctor_first_name, d.last_name AS doctor_last_name 
        FROM reports r
        JOIN doctors d ON r.doctor_id = d.id
        WHERE r.patient_id = %s
        ORDER BY r.created_at DESC
    """, (patient_id,))
    reports = cur.fetchall()

    return render_template('patient_dashboard.html', reports=reports, appointments=appointments, report_search_query=None, appointment_search_query=query)


#end of patirnt dashboard............................................................................................................................................................................................................................

# Predict Diabetes...................................................................................................................................................................................................................................
# @app.route('/predict_diabetes', methods=['POST'])
# @login_required
# def predict_diabetes():
#     if current_user.role != 'nurse':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     # Gather input data from the form
#     age = request.form['age']
#     bmi = request.form['bmi']
#     glucose = request.form['glucose']
#     patient_id = request.form['patient_id']

#     # Run the prediction using the model
#     prediction = model.predict([[age, bmi, glucose]])

#     # Store the prediction in the database for the doctor to review
#     cur = mysql.connection.cursor()
#     cur.execute("INSERT INTO predictions (patient_id, prediction, type) VALUES (%s, %s, 'diabetes')", 
#                 (patient_id, prediction[0]))
#     mysql.connection.commit()

#     flash('Diabetes prediction stored successfully!')
#     return redirect(url_for('nurse_dashboard'))

# Book Appointment...................................................................................................................................................................................................................................
@app.route('/book_appointment', methods=['POST'])
@login_required
def book_appointment():
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    # Get data from the form
    doctor_id = request.form['doctor_id']
    appointment_date = request.form['appointment_date']
    patient_id = request.form['patient_id']

    # Insert appointment into the database
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO appointments (patient_id, doctor_id, appointment_date) VALUES (%s, %s, %s)", 
                (patient_id, doctor_id, appointment_date))
    mysql.connection.commit()

    flash('Appointment booked successfully!')
    return redirect(url_for('nurse_dashboard'))

@app.route('/update_patient/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def update_patient(patient_id):
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Fetch the patient's current details
    if request.method == 'GET':
        cur.execute("SELECT * FROM patients WHERE id = %s AND nurse_id = (SELECT id FROM nurses WHERE user_id = %s)", (patient_id, current_user.id))
        patient = cur.fetchone()

        if not patient:
            flash('Patient not found or you do not have permission to edit this patient.')
            return redirect(url_for('nurse_dashboard'))

        # Render the form with the patient's current details
        return render_template('update_patient.html', patient=patient)

    # If the form is submitted, update the patient's details
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        gender = request.form['gender']
        contact_number = request.form['contact_number']

        cur.execute("""
            UPDATE patients 
            SET first_name = %s, last_name = %s, age = %s, gender = %s, contact_number = %s
            WHERE id = %s AND nurse_id = (SELECT id FROM nurses WHERE user_id = %s)
        """, (first_name, last_name, age, gender, contact_number, patient_id, current_user.id))
        mysql.connection.commit()

        flash('Patient updated successfully!')
        return redirect(url_for('nurse_dashboard'))
#Delete patient...................................................................................................................................................................................................................................    
@app.route('/delete_patient/<int:patient_id>')
@login_required
def delete_patient(patient_id):
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Delete the patient, but ensure the nurse has permission to delete this patient
    cur.execute("DELETE FROM patients WHERE id = %s AND nurse_id = (SELECT id FROM nurses WHERE user_id = %s)", (patient_id, current_user.id))
    mysql.connection.commit()

    flash('Patient deleted successfully!')
    return redirect(url_for('nurse_dashboard'))

@app.route('/make_appointment/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def make_appointment(patient_id):
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Fetch the list of specializations
    cur.execute("SELECT DISTINCT specialization FROM doctors")
    specializations = [row[0] for row in cur.fetchall()]

    # Fetch the doctors list for the appointment form
    cur.execute("SELECT id, first_name, last_name, specialization FROM doctors")
    doctors = cur.fetchall()

    # If GET request, display the form
    if request.method == 'GET':
        # Fetch the patient information to show in the form
        cur.execute("SELECT * FROM patients WHERE id = %s AND nurse_id = (SELECT id FROM nurses WHERE user_id = %s)", (patient_id, current_user.id))
        patient = cur.fetchone()

        if not patient:
            flash('Patient not found or you do not have permission to book an appointment for this patient.')
            return redirect(url_for('nurse_dashboard'))

        return render_template('make_appointment.html', patient=patient, doctors=doctors)

    # If POST request, create the appointment
    if request.method == 'POST':
        doctor_id = request.form['doctor_id']
        appointment_date = request.form['appointment_date']
        appointment_time = request.form['appointment_time']
        reason = request.form['reason']

        # Insert the new appointment into the appointments table
        cur.execute(
            "INSERT INTO appointments (patient_id, doctor_id, appointment_date, appointment_time, reason) VALUES (%s, %s, %s, %s, %s)",
            (patient_id, doctor_id, appointment_date, appointment_time, reason)
        )
        mysql.connection.commit()
        cur.close()

        flash('Appointment made successfully!')
        return redirect(url_for('nurse_dashboard'))
    
#Search patient...................................................................................................................................................................................................................................    
@app.route('/search_patient', methods=['POST'])
@login_required
def search_patient():
    if current_user.role != 'doctor':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    search_query = request.form['search_query']

    cur = mysql.connection.cursor()

    # Search for patients by name (first_name or last_name)
    cur.execute("""
        SELECT id, first_name, last_name, age, contact_number 
        FROM patients
        WHERE first_name LIKE %s OR last_name LIKE %s
    """, (f'%{search_query}%', f'%{search_query}%'))
    
    patients = cur.fetchall()

    # Debugging: Print out the entire patient data structure
    for patient in patients:
        print(f"Patient Data: {patient}")  # This will print each patient's row, ensuring patient[0] is the ID

    # Fetch the doctor's appointments for display in the dashboard
    cur.execute("""
        SELECT 
            a.appointment_date, 
            a.appointment_time, 
            a.reason, 
            p.first_name, 
            p.last_name, 
            p.age, 
            p.contact_number 
        FROM appointments a
        JOIN patients p ON a.patient_id = p.id
        WHERE a.doctor_id = (SELECT id FROM doctors WHERE user_id = %s)
        ORDER BY a.appointment_date, a.appointment_time
    """, (current_user.id,))
    appointments = cur.fetchall()

    return render_template('doctor_dashboard.html', patients=patients, appointments=appointments)

#Upload Reports......................................................................................................................Original........................................................................................................................
# # Define where to save uploaded files (e.g., reports)
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# # Ensure the upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Utility function to check if the uploaded file type is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload_report', methods=['POST'])
# @login_required
# def upload_report():
#     if current_user.role != 'nurse':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     if 'report_file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['report_file']
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)

#     if file and allowed_file(file.filename):
#         # Secure the filename and save the file
#         filename = secure_filename(file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)

#         # Get form data
#         patient_id = request.form['patient_id']
#         doctor_id = request.form['doctor_id']
#         report_name = request.form['report_name']

#          # Fetch the nurse_id based on the current user's user_id
#         cur = mysql.connection.cursor()
#         cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
#         nurse_data = cur.fetchone()

#         if not nurse_data:
#             flash('Nurse not found!')
#             return redirect(url_for('nurse_dashboard'))

#         nurse_id = nurse_data[0]  # Extract nurse_id

#          # Insert the report details into the database
#         cur.execute("""
#             INSERT INTO reports (report_name, file_path, patient_id, nurse_id, doctor_id) 
#             VALUES (%s, %s, %s, %s, %s)
#         """, (report_name, file_path, patient_id, nurse_id, doctor_id))
#         mysql.connection.commit()
#         cur.close()

#         flash('Report uploaded successfully!')
#         return redirect(url_for('nurse_dashboard'))

#     flash('Invalid file type. Please upload a PDF or an image.')
#     return redirect(url_for('nurse_dashboard'))

#### Edited Upload Report ........................................................................*.*.*.................................*.*.*.....................
# Define where to save uploaded files (e.g., reports)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Utility function to check if the uploaded file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_report', methods=['POST'])
@login_required
def upload_report():
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    if 'report_file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['report_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get form data
        patient_id = request.form['patient_id']
        doctor_id = request.form['doctor_id']
        report_name = request.form['report_name']

         # Fetch the nurse_id based on the current user's user_id
        cur = mysql.connection.cursor()
        cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
        nurse_data = cur.fetchone()

        if not nurse_data:
            flash('Nurse not found!')
            return redirect(url_for('nurse_dashboard'))

        nurse_id = nurse_data[0]  # Extract nurse_id

         # Insert the report details into the database
        cur.execute("""
            INSERT INTO reports (report_name, file_path, patient_id, nurse_id, doctor_id) 
            VALUES (%s, %s, %s, %s, %s)
        """, (report_name, file_path, patient_id, nurse_id, doctor_id))
        mysql.connection.commit()
        cur.close()

        flash('Report uploaded successfully!')
        return redirect(url_for('nurse_dashboard'))

    flash('Invalid file type. Please upload a PDF or an image.')
    return redirect(url_for('nurse_dashboard'))

# Nurse and Doctor Shared Report Upload Route.................................................................*..*.*........................*.*.*.*.....................
# @app.route('/upload_report/<int:patient_id>', methods=['GET', 'POST'])
# @login_required
# def upload_report(patient_id):
#     if current_user.role not in ['nurse', 'doctor']:
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     if request.method == 'POST':
#         if 'report_file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)

#         file = request.files['report_file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             report_name = request.form['report_name']

#             # Get nurse_id or doctor_id based on the current user's role
#             cur = mysql.connection.cursor()

#             if current_user.role == 'nurse':
#                 cur.execute("SELECT id FROM nurses WHERE user_id = %s", (current_user.id,))
#                 nurse_data = cur.fetchone()
#                 nurse_id = nurse_data[0] if nurse_data else None
#                 doctor_id = None
#             elif current_user.role == 'doctor':
#                 cur.execute("SELECT id FROM doctors WHERE user_id = %s", (current_user.id,))
#                 doctor_data = cur.fetchone()
#                 doctor_id = doctor_data[0] if doctor_data else None
#                 nurse_id = None

#             # Insert the report into the database
#             cur.execute("""
#                 INSERT INTO reports (report_name, file_path, patient_id, nurse_id, doctor_id) 
#                 VALUES (%s, %s, %s, %s, %s)
#             """, (report_name, file_path, patient_id, nurse_id, doctor_id))
#             mysql.connection.commit()
#             cur.close()

#             flash('Report uploaded successfully!')
#             return redirect(url_for('nurse_dashboard') if current_user.role == 'nurse' else url_for('doctor_dashboard'))

#     # Fetch patient details to display on the upload form (if needed)
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT first_name, last_name FROM patients WHERE id = %s", (patient_id,))
#     patient = cur.fetchone()

#     return render_template('upload_report.html', patient=patient)
# New route for doctor to upload a report from their dashboard without a specific patient_id.......................................*.*.*................................*.*.*.*...................................
# @app.route('/doctor_upload_report', methods=['GET', 'POST'])
# @login_required
# def doctor_upload_report():
#     if current_user.role != 'doctor':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     if request.method == 'POST':
#         if 'report_file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)

#         file = request.files['report_file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)

#             report_name = request.form['report_name']
#             patient_id = request.form['patient_id']  # Select patient ID from the dropdown

#             # Get doctor_id for the current user
#             cur = mysql.connection.cursor()
#             cur.execute("SELECT id FROM doctors WHERE user_id = %s", (current_user.id,))
#             doctor_data = cur.fetchone()
#             doctor_id = doctor_data[0] if doctor_data else None

#             # Insert the report into the database
#             cur.execute("""
#                 INSERT INTO reports (report_name, file_path, patient_id, nurse_id, doctor_id) 
#                 VALUES (%s, %s, %s, NULL, %s)
#             """, (report_name, file_path, patient_id, doctor_id))
#             mysql.connection.commit()
#             cur.close()

#             flash('Report uploaded successfully!')
#             return redirect(url_for('doctor_dashboard'))

#     # Fetch all patients to allow the doctor to select a patient
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT id, first_name, last_name FROM patients")
#     patients = cur.fetchall()

    # return render_template('doctor_upload_report.html', patients=patients)



#View reports..........................................................................................................................................................................................................................................
@app.route('/view_reports/<int:patient_id>')
@login_required
def view_reports(patient_id):
    if current_user.role not in ['nurse', 'doctor', 'patient']:
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    # Fetch reports for the specified patient
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT r.report_name, r.file_path, r.created_at, n.first_name, n.last_name, d.first_name, d.last_name
        FROM reports r
        JOIN nurses n ON r.nurse_id = n.id
        JOIN doctors d ON r.doctor_id = d.id
        WHERE r.patient_id = %s
    """, (patient_id,))
    reports = cur.fetchall()
    cur.close()

    return render_template('view_reports.html', reports=reports)

#Delete reports..........................................................................................................................................................................................................................................
@app.route('/delete_report/<int:report_id>', methods=['GET'])
@login_required
def delete_report(report_id):
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Delete the report from the database
    cur.execute("DELETE FROM reports WHERE id = %s", (report_id,))
    mysql.connection.commit()
    cur.close()

    flash('Report deleted successfully!')
    return redirect(url_for('nurse_dashboard'))

#Edit report..........................................................................................................................................................................................................................................
@app.route('/edit_report/<int:report_id>', methods=['GET', 'POST'])
@login_required
def edit_report(report_id):
    if current_user.role != 'nurse':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        # Get the updated form data
        report_name = request.form['report_name']
        file = request.files['report_file']

        # If a new file is uploaded, save the file
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Update the report details along with the file path
            cur.execute("""
                UPDATE reports 
                SET report_name = %s, file_path = %s
                WHERE id = %s
            """, (report_name, file_path, report_id))
        else:
            # Update the report name only
            cur.execute("""
                UPDATE reports 
                SET report_name = %s
                WHERE id = %s
            """, (report_name, report_id))

        mysql.connection.commit()
        cur.close()

        flash('Report updated successfully!')
        return redirect(url_for('nurse_dashboard'))

    # Fetch the current report details for editing
    cur.execute("SELECT * FROM reports WHERE id = %s", (report_id,))
    report = cur.fetchone()

    if not report:
        flash('Report not found!')
        return redirect(url_for('nurse_dashboard'))

    cur.close()

    return render_template('edit_report.html', report=report)
      
        
from flask import send_from_directory
#Upload..........................................................................................................................................................................................................................................
# @app.route('/uploads/<filename>')
# @login_required
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
#Admin Dashboard...................................................................................................................................................................................................................................
@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    # Ensure the user is an admin
    if current_user.role != 'admin':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Fetch all users (patients, doctors, nurses, admins)
    cur.execute("""
        SELECT id, username, email, role, created_at FROM users
    """)
    users = cur.fetchall()

    return render_template('admin_dashboard.html', users=users)

#Edit User...................................................................................................................................................................................................................................
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if current_user.role != 'admin':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        role = request.form['role']

        cur.execute("""
            UPDATE users 
            SET username = %s, password = %s, email = %s, role = %s 
            WHERE id = %s
        """, (username, password, email, role, user_id))

        mysql.connection.commit()
        flash('User updated successfully!')
        return redirect(url_for('admin_dashboard'))

    # Fetch user details for editing
    cur.execute("""
        SELECT id, username, password, email, role FROM users WHERE id = %s
    """, (user_id,))
    user = cur.fetchone()

    return render_template('edit_user.html', user=user)

#Delete User...................................................................................................................................................................................................................................
@app.route('/delete_user/<int:user_id>')
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Delete user
    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
    mysql.connection.commit()

    flash('User deleted successfully!')
    return redirect(url_for('admin_dashboard'))
#add prescription...................................................................................................................................................................................................................................
# @app.route('/add_prescription/<int:patient_id>', methods=['GET', 'POST'])
# @login_required
# def add_prescription(patient_id):
#     # Ensure the user is a doctor
#     if current_user.role != 'doctor':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     cur = mysql.connection.cursor()

#     # Fetch the patient details
#     cur.execute("""
#         SELECT id, first_name, last_name FROM patients WHERE id = %s
#     """, (patient_id,))
#     patient = cur.fetchone()

#     if request.method == 'POST':
#         diagnosis = request.form['diagnosis']
#         prescription = request.form['prescription']
        
#         # Insert the prescription into the reports table
#         cur.execute("""
#             INSERT INTO reports (patient_id, doctor_id, diagnosis, prescription)
#             VALUES (%s, %s, %s, %s)
#         """, (patient_id, current_user.id, diagnosis, prescription))

#         mysql.connection.commit()
#         flash('Prescription added successfully!')
#         return redirect(url_for('doctor_dashboard'))

#     return render_template('add_prescription.html', patient=patient)

# # Define the folder for storing uploads
# UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # This will create 'uploads' folder in the current working directory

# # Ensure the folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # Add it to Flask config
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Initialize Flask app
# app = Flask(__name__)

# # Set upload folder and allowed file extensions
# UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
# ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

# # Ensure the upload folder exists
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Utility function to check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Edit Appointment:........................................................................................................................................................................................................................

# @app.route('/edit_appointment/<int:appointment_id>', methods=['GET', 'POST'])
# @login_required
# def edit_appointment(appointment_id):
#     if current_user.role != 'nurse':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     cur = mysql.connection.cursor()

#     if request.method == 'POST':
#         appointment_date = request.form['appointment_date']
#         appointment_time = request.form['appointment_time']
#         reason = request.form['reason']

#         cur.execute("""
#             UPDATE appointments 
#             SET appointment_date = %s, appointment_time = %s, reason = %s 
#             WHERE id = %s
#         """, (appointment_date, appointment_time, reason, appointment_id))
#         mysql.connection.commit()

#         flash('Appointment updated successfully!')
#         return redirect(url_for('nurse_dashboard'))

#     # Fetch current appointment details
#     cur.execute("SELECT * FROM appointments WHERE id = %s", (appointment_id,))
#     appointment = cur.fetchone()

#     return render_template('edit_appointment.html', appointment=appointment)




#Delete Appointment..

# @app.route('/delete_appointment/<int:appointment_id>')
# @login_required
# def delete_appointment(appointment_id):
#     if current_user.role != 'nurse':
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     cur = mysql.connection.cursor()

#     # Delete the appointment
#     cur.execute("DELETE FROM appointmets WHERE id = %s AND nurse_id = (SELECT id FROM nurses WHERE user_id = %s)", (appointment_id, current_user.id))
#     mysql.connection.commit()

#     flash('Appointment deleted successfully!')
#     return redirect(url_for('nurse_dashboard'))


#patient Profile.................................................................................................................................................................................................................................................

# @app.route('/patient_profile/<int:patient_id>')
# @login_required
# def patient_profile(patient_id):
#     # Check if the user has access to this patient (nurse, doctor, or the patient themselves)
#     if current_user.role not in ['nurse', 'doctor', 'patient']:
#         flash('Unauthorized access!')
#         return redirect(url_for('home'))

#     cur = mysql.connection.cursor()

#     # Fetch the patient details
#     cur.execute("SELECT * FROM patients WHERE id = %s", (patient_id,))
#     patient = cur.fetchone()

#     if not patient:
#         flash('Patient not found!')
#         return redirect(url_for('home'))

#     # Fetch the patient's reports
#     cur.execute("""
#         SELECT r.report_name, r.file_path, r.created_at, 
#                n.first_name AS nurse_first_name, n.last_name AS nurse_last_name,
#                d.first_name AS doctor_first_name, d.last_name AS doctor_last_name
#         FROM reports r
#         JOIN nurses n ON r.nurse_id = n.id
#         JOIN doctors d ON r.doctor_id = d.id
#         WHERE r.patient_id = %s
#         ORDER BY r.created_at DESC
#     """, (patient_id,))
#     reports = cur.fetchall()

#     # Fetch the patient's appointments
#     cur.execute("""
#         SELECT a.appointment_date, a.appointment_time, a.reason, 
#                d.first_name AS doctor_first_name, d.last_name AS doctor_last_name
#         FROM appointments a
#         JOIN doctors d ON a.doctor_id = d.id
#         WHERE a.patient_id = %s
#         ORDER BY a.appointment_date, a.appointment_time
#     """, (patient_id,))
#     appointments = cur.fetchall()

#     cur.close()

#     return render_template('patient_profile.html', patient=patient, reports=reports, appointments=appointments)


