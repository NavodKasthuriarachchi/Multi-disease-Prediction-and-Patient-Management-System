from flask import render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, current_user, login_required
from project_app import app
from project_app.__init__ import mysql
from project_app.models import User
import os
import numpy as np 
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from werkzeug.utils import secure_filename
from flask import send_from_directory
from MySQLdb.cursors import DictCursor

# Home route
@app.route('/')
@login_required
def home():
    return render_template('home.html')

# Load the diabetes prediction model.....................................................................................................................................................................................................................
with open("J:/Final Project/multi_disease_system/project_app/diabetes.pkl", 'rb') as model_file:
    diabetes_model = pickle.load(model_file)

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

            # Make prediction
            prediction = diabetes_model.predict(input_data)

            # Interpret the prediction result
            prediction_result = "likely" if prediction[0] == 1 else "unlikely"

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
#Register Nurse...................................................................................................................................................................................................................................
@app.route('/register_nurse', methods=['GET', 'POST'])
def register_nurse():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        contact_number = request.form['contact_number']
        
        # Insert user into `users` table
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'nurse')", 
                    (username, password, email))
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
        
        # Insert user into `users` table
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password, email, role) VALUES (%s, %s, %s, 'doctor')", 
                    (username, password, email))
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
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# .................................................................................................................................Load your prediction model (e.g., for diabetes)
# model = pickle.load(open('diabetes_model.pkl', 'rb'))

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

    nurse_id = nurse_data[0]  # Extract the nurse's id

    # Fetch the patients registered by this nurse
    cur.execute("SELECT * FROM patients WHERE nurse_id = %s", (nurse_id,))
    patients = cur.fetchall()

    # Pass the patients data to the template
    return render_template('nurse_dashboard.html', patients=patients)


# Doctor Dashboard...................................................................................................................................................................................................................................
@app.route('/doctor_dashboard')
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
        # WHERE p.doctor_id = (SELECT id FROM doctors WHERE user_id = %s)
        # ORDER BY pr.predicted_at DESC
    """, (current_user.id,))
    
    predictions = cur.fetchall()

    cur.close()

    return render_template('doctor_dashboard.html', appointments=appointments, predictions=predictions)


from MySQLdb.cursors import DictCursor
#Patient Dashboard...................................................................................................................................................................................................................................
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

    # Fetch patient's prescriptions
    cur.execute("""
        SELECT r.diagnosis, r.prescription, r.created_at, d.first_name AS doctor_first_name, d.last_name AS doctor_last_name 
        FROM reports r
        JOIN doctors d ON r.doctor_id = d.id
        WHERE r.patient_id = %s
        ORDER BY r.created_at DESC
    """, (patient_id,))
    prescriptions = cur.fetchall()

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

    return render_template('patient_dashboard.html', prescriptions=prescriptions, appointments=appointments)





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

    # Fetch the doctors list for the appointment form
    cur.execute("SELECT id, first_name, last_name FROM doctors")
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


# Define where to save uploaded files (PDFs or images)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the uploaded file is allowed (PDF or image)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#upload Report...................................................................................................................................................................................................................................
@app.route('/upload_report/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def upload_report(patient_id):
    # Debugging: Print the received patient_id
    print(f"Received patient_id: {patient_id}")  # This should be an integer

    if current_user.role != 'doctor':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Fetch patient details
    if request.method == 'GET':
        cur.execute("SELECT first_name, last_name FROM patients WHERE id = %s", (patient_id,))
        patient = cur.fetchone()

        if not patient:
            flash('Patient not found.')
            return redirect(url_for('doctor_dashboard'))

        return render_template('upload_report.html', patient=patient)


    # If POST request, handle file upload
    if request.method == 'POST':
        # Check if the file is in the request
        if 'report_file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['report_file']

        # If no file is selected
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Check if the file is allowed (PDF or image)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Determine the file type (PDF or image)
            file_type = 'pdf' if filename.lower().endswith('pdf') else 'image'

            # Store the file path in the new report_uploads table
            cur.execute("""
                INSERT INTO report_uploads (patient_id, doctor_id, file_path, file_type)
                VALUES (%s, (SELECT id FROM doctors WHERE user_id = %s), %s, %s)
            """, (patient_id, current_user.id, file_path, file_type))
            mysql.connection.commit()
            cur.close()

            flash('Report uploaded successfully!')
            return redirect(url_for('doctor_dashboard'))

        else:
            flash('Invalid file format. Please upload a PDF or an image.')
            return redirect(request.url)
        
        
from flask import send_from_directory
#Upload..........................................................................................................................................................................................................................................
@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
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
@app.route('/add_prescription/<int:patient_id>', methods=['GET', 'POST'])
@login_required
def add_prescription(patient_id):
    # Ensure the user is a doctor
    if current_user.role != 'doctor':
        flash('Unauthorized access!')
        return redirect(url_for('home'))

    cur = mysql.connection.cursor()

    # Fetch the patient details
    cur.execute("""
        SELECT id, first_name, last_name FROM patients WHERE id = %s
    """, (patient_id,))
    patient = cur.fetchone()

    if request.method == 'POST':
        diagnosis = request.form['diagnosis']
        prescription = request.form['prescription']
        
        # Insert the prescription into the reports table
        cur.execute("""
            INSERT INTO reports (patient_id, doctor_id, diagnosis, prescription)
            VALUES (%s, %s, %s, %s)
        """, (patient_id, current_user.id, diagnosis, prescription))

        mysql.connection.commit()
        flash('Prescription added successfully!')
        return redirect(url_for('doctor_dashboard'))

    return render_template('add_prescription.html', patient=patient)


