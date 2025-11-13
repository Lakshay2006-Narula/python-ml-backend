# In extensions.py

from flask_sqlalchemy import SQLAlchemy

# Create the database instance, but don't
# connect it to the app yet.
db = SQLAlchemy()