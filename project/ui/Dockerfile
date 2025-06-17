# project/ui/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Create a slim requirements.txt on the fly for the UI
# This avoids copying the entire pyproject.toml with all the heavy ML deps
RUN echo "streamlit" > requirements.txt && \
    echo "pandas" >> requirements.txt && \
    echo "requests" >> requirements.txt

# Install only the UI dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the UI application code
# Assuming you place the ui app.py file inside project/ui/
COPY project/ui/app.py .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.headless", "true"]