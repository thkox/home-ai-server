# Home AI Server

<p align="center">
  <img src="./images/app_logo-cropped.png" alt="app logo" width="150" height="150">
</p>

**Home AI Server** provides the backend infrastructure for the Home AI system. It manages user data, conversations, and the interaction with Large Language Models (LLMs). The server runs locally, ensuring that all user data stays within the local network for maximum privacy.

## Features

- Role-Based Access Control (RBAC) for managing user permissions.
- Local data storage using PostgreSQL.
- API-based communication with the Android client.
- Document analysis and embedding storage using ChromaDB.
- Interfacing with LLMs for natural language processing using Ollama.

## Requirements

- Operating System: Windows, macOS, or Linux.
- Processor: Quad-core CPU (Intel or AMD).
- Memory: 16 GB RAM.
- Storage: 50 GB free disk space.
- Dependencies:
  - Docker & Docker Compose
  - PostgreSQL
  - Ollama (for LLM interactions)

## Technologies Used

- **FastAPI**: Web framework for building the server API.
- **PostgreSQL**: Relational database for storing user data, conversations, and document metadata.
- **ChromaDB**: For managing document embeddings used in response generation.
- **Langchain**: For handling memory in LLM interactions.
- **Ollama**: For running local LLMs.
- **Docker**: Containerization for easy deployment and management.

## Installation

1. **Install Dependencies**:
   - Install Docker: [Docker Installation Guide](https://docs.docker.com/engine/install/)
   - Install PostgreSQL: [PostgreSQL Installation Guide](https://www.postgresql.org/download/)
   - Install Ollama: [Ollama Installation Guide](https://ollama.com/download)

2. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/home-ai-server.git
   cd home-ai-server
   ```

3. **Set Up Environment Variables**:
   Edit the `.env` file with your desired configuration.

4. **Build and Run the Docker Container**:
   ```bash
   docker-compose build
   docker-compose up
   ```

   The server will start on the default port `8000`. You can access the API at `http://localhost:8000`.

## API Endpoints

- **Root**:
  - `GET /`: Read Root

- **Authentication**:
  - `POST /auth/register`: Register a new user.
  - `POST /auth/login`: Log in and receive a JWT token.

- **Users**:
  - `PUT /users/me/password`: Change the current user's password.
  - `PUT /users/{user_id}/password`: Change any user's password (admin only).
  - `GET /users/me/details`: Get authenticated user's details.
  - `PUT /users/me/profile`: Update the authenticated user's profile.
  - `PUT /users/{user_id}/profile`: Update any user's profile (admin only).

- **Conversations**:
  - `POST /conversations`: Start a new conversation with the LLM.
  - `GET /conversations/{conversation_id}`: Get a specific conversation by ID.
  - `GET /conversations/me`: Get all conversations of the authenticated user.
  - `DELETE /conversations/{conversation_id}`: Delete a conversation by ID.
  - `GET /conversations/{conversation_id}/messages`: Get messages of a specific conversation.
  - `POST /conversations/{conversation_id}/continue`: Continue an existing conversation.

- **Documents**:
  - `POST /documents/upload`: Upload a document for analysis.
  - `GET /documents/{document_id}`: Get document details by ID.
  - `GET /documents/me`: Retrieve all documents uploaded by the authenticated user.
  - `DELETE /documents/{document_id}`: Delete a document by ID.

## Database Schema

The server uses PostgreSQL to store user profiles, conversations, and document metadata. Here's an overview of the main tables:

- **Users**: Stores user profiles and credentials.
- **Conversations**: Stores the conversation history of each user.
- **Documents**: Stores metadata of uploaded documents.

## Future Enhancements

- Containerize ChromaDB for better separation of concerns.
- Support for additional file types and enhanced file processing.
- Integration with home automation systems (e.g., Home Assistant) to control smart devices using voice commands.

## Troubleshooting

- **Server not starting**: Check the logs with `docker-compose logs`.
- **Database connection issues**: Ensure PostgreSQL is running and correctly configured in the `.env` file.
