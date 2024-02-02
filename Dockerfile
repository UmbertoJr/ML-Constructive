# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container to /app
WORKDIR /app

# Copy the version1.1 directory contents into the container at /app
COPY version1.1/ /app/


# Install OpenGL library
RUN apt-get update
RUN apt-get install libgl1-mesa-glx


# update pip install
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/version1/requirements.txt

# Install PyConcorde to generate optimal TSP tours
RUN pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'

# Install OpenCV
RUN pip install opencv-python

# Install torchsummary
RUN pip install torchsummary
