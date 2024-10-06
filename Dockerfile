# Use the official Python image with a slim variant
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set environment variables with default values (can be overridden)
ENV DATABASE_URL=postgres
ENV DATABASE_USERNAME=postgres
ENV DATABASE_PASSWORD=password1234
ENV DATABASE_NAME=homeai
ENV CHROMADB_PERSIST_DIRECTORY=/data/chroma_db
ENV DOCUMENTS_DIRECTORY=/data/documents
ENV ALGORITHM=HS256
ENV ACCESS_TOKEN_EXPIRE_MINUTES=259200
ENV OLLAMA_URL=http://ollama:11434/
ENV MODEL_NAME=llama3.1:8b-instruct-q4_1
ENV EMBEDDING_MODEL_NAME=nomic-embed-text
ENV PORT=8000

# Expose the port (default 8000)
EXPOSE ${PORT}

# Create directories for data storage
RUN mkdir -p ${CHROMADB_PERSIST_DIRECTORY} ${DOCUMENTS_DIRECTORY}

# Set the PYTHONPATH to include the /app directory
ENV PYTHONPATH=/app

# Set the command to run the app and print the IP and port
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]