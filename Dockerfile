# Start with an NVIDIA CUDA image that includes the necessary CUDA drivers and libraries
FROM nvidia/cuda:12.3.1-base-ubuntu22.04
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
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt-get update
RUN sudo apt-get install libglib2.0-0


# Update pip and install the requirements
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde' \
    && pip install opencv-python \
    && pip install torchsummary
