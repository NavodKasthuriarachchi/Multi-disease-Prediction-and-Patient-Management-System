import sys
import os
from flask import Flask, render_template
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from project_app import app

if __name__ == '__main__':
    app.run(debug=True)
