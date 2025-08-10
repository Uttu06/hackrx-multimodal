# Step 1: Base Image
# Start with a Python 3.10 image on Debian "Bookworm" for good package support.
# Using the "slim" variant keeps the image size smaller.
FROM python:3.10-slim-bookworm

# Step 2: Set Environment Variables
# These are recommended for running Python in a containerized environment.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Step 3: Install System Dependencies
# This is the crucial step for Tesseract OCR. We update the package list and install it.
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Step 4: Set the Working Directory
# All subsequent commands will be run from this directory inside the container.
WORKDIR /app

# Step 5: Install Python Dependencies
# First, copy only the requirements file to leverage Docker's build cache.
# If this file doesn't change, Docker won't re-run this step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy Project Source Code
# Copy all your project files (src folder, main.py, etc.) into the working directory.
# This is done after pip install, so changes to code don't invalidate the dependency cache.
COPY . .

# Step 7: Expose Port
# Your application is a FastAPI web server. We expose port 8000, which is the default for Uvicorn.
EXPOSE 8000

# Step 8: Define Startup Command
# This command will be executed when the container starts. It runs the FastAPI app using uvicorn.
# --host 0.0.0.0 is essential to make the app accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]