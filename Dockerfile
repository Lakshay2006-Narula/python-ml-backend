# 1. Use the base image
FROM python:3.11.0-slim

# 2. Set the working directory
WORKDIR /app

# -------------------------------------------------------
# 3. INSTALL SYSTEM DEPENDENCIES
# -------------------------------------------------------
# We need pkg-config, gcc, default-libmysqlclient-dev for mysql
# We need libgl1 and libglib2.0-0 because you are using PyQt5/Opencv related libs
RUN apt-get update && apt-get install -y \
    pkg-config \
    gcc \
    default-libmysqlclient-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python packages
# We upgrade pip first, then install with a high timeout (1000s) to prevent network errors
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 6. Copy the rest of the code
COPY . .

# 7. (Optional) Set environment variables
ENV FLASK_APP=app.py

# 8. Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
