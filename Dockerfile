# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (e.g., for pandas, scikit-learn)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "affordable_housing.api.main:app", "--host", "0.0.0.0", "--port", "8000"]