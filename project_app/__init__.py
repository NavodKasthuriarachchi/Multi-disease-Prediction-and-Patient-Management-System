import os
from flask import Flask
from flask_mysqldb import MySQL
from flask_login import LoginManager
from project_app.models import User

# Correct path for static and template folders
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

app.config['SECRET_KEY'] = '9791'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '9791'
app.config['MYSQL_DB'] = 'patient_management_system'

mysql = MySQL(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cur.fetchone()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2], user_data[4])
    return None

from project_app import routes
