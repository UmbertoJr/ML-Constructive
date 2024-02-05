# Start with an NVIDIA CUDA image that includes the necessary CUDA drivers and libraries
FROM nvidia/cuda:12.3.1-base-ubuntu22.04
# Note: The base image already includes CUDA, so there's no need to install CUDA or NVIDIA drivers manually

# Use a label to maintain metadata
LABEL maintainer="Your Name <your.email@example.com>"

# Set the working directory in the container to /app
WORKDIR /app

# Copy the version1.1 directory contents into the container at /app
COPY version1.1/ /app/

# Install system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Update pip and install the requirements
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde' \
    && pip install opencv-python \
    && pip install torchsummary




FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04

# Install Python 3 and pip
RUN apt-get update && apt-get install -y python3 python3-pip libgl1-mesa-glx

# Install PyTorch
RUN pip3 install torch torchvision

# Add your application files
COPY . /app
WORKDIR /app

# update pip install
RUN pip install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyConcorde to generate optimal TSP tours
RUN pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'

# Install OpenCV
RUN pip install opencv-python

# Install torchsummary
RUN pip install torchsummary