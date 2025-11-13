# Use the same Python version you specified in render.yaml
FROM python:3.11.0-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Set the working directory in the container
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Gunicorn start command from your original render.yaml
# Render will automatically set the $PORT variable
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 4 --timeout 300 --log-level info