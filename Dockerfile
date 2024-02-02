# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR ./version1

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ./version1/requirements.txt

# Install PyConcorde to generate optimal TSP tours
RUN pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'

# Make port 80 available to the world outside this container
EXPOSE 80

