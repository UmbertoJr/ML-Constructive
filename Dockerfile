# Start with an NVIDIA CUDA image that includes the necessary CUDA drivers and libraries
FROM nvcr.io/nvidia/pytorch:23.12-py3
# Note: The base image already includes CUDA, so there's no need to install CUDA or NVIDIA drivers manually

# Use a label to maintain metadata
LABEL maintainer="Umberto Junior Mele <u.mele.coding@gmail.com>"

# Set the working directory in the container to /app
WORKDIR /app

# Copy the version1.1 directory contents into the container at /app
COPY version1.1/ /app/

# Install git
RUN apt-get update && apt-get install -y git

# Install system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    python3-pip \
    python3.8 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx

# Update pip and install the requirements
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde' \
    && pip install opencv-python \
    && pip install torchsummary
