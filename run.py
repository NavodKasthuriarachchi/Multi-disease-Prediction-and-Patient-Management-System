import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from project_app import app

if __name__ == '__main__':
    app.run(debug=True)
