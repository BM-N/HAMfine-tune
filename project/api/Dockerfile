# project/api/Dockerfile
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the files needed for dependency installation first
COPY pyproject.toml ./

# Copy your entire project source code
# This allows the package to be installed
COPY project/ ./project/

# Install the project itself, along with all dependencies from pyproject.toml
# This makes `from project.models...` work
RUN pip install .

# Copy artifacts that are not part of the source code package
COPY best_model.pth .

# The command to run the API when the container launches
# Note the module path now points to the file within the installed package
CMD ["uvicorn", "project.api.main:app", "--host", "0.0.0.0", "--port", "8000"]