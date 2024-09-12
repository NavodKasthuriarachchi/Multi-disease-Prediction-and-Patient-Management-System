from flask import Flask
from flask_mysqldb import MySQL
from flask_login import LoginManager
from project_app.models import User  # Import your User model
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'))

app.config['SECRET_KEY'] = '9791'

# Configuration for MySQL connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'  # Replace with your MySQL username
app.config['MYSQL_PASSWORD'] = '9791'  # Replace with your MySQL password
app.config['MYSQL_DB'] = 'patient_management_system'

# Initialize MySQL
mysql = MySQL(app)

# Setup Flask-Login for user management
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User loader for Flask-Login
from project_app.models import User

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cur.fetchone()

    if user_data:
        return User(user_data[0], user_data[1], user_data[2], user_data[4])  # Correct role index
  # Adjust based on your user table structure
    return None 

from project_app import routes  # Importing routes to keep the structure clean
