# docker build -t benchmark:latest .

# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /benchmark

# Copy the current directory contents into the container at /benchmark
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run script.py when the container launches
ENTRYPOINT ["python", "benchmark.py"]
