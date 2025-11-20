# 1. Use the base image
FROM python:3.11.0-slim

# 2. Set the working directory
WORKDIR /app

# -------------------------------------------------------
# 3. INSTALL SYSTEM DEPENDENCIES (The Fix)
# -------------------------------------------------------
# We need pkg-config, gcc, and libmysqlclient-dev to compile mysqlclient
RUN apt-get update && apt-get install -y \
    pkg-config \
    gcc \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the code
COPY . .

# 7. (Optional) Set environment variables
ENV FLASK_APP=app.py

# 8. Command to run the app (Adjust 'app:app' if your main file is named differently)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
