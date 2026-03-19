FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Step 1: Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: Copy your code and weights
COPY . /app

# Step 3: Set the entrypoint to use python3
ENTRYPOINT ["python3", "main.py"]
